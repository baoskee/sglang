from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from functools import partial
from queue import Empty
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.mem_cache.unified_cache_components import (
    _NUM_COMPONENT_TYPES,
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentData,
    ComponentType,
    EvictLayer,
    FullComponent,
    MambaComponent,
    SharedAnchorComponent,
    SWAComponent,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.mem_cache.utils import convert_to_bigram_key
from sglang.srt.session.streaming_session import StreamingSession

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


class UnifiedTreeNode:
    counter = 0

    def __init__(self, tree_components: tuple[ComponentType, ...]):
        self.children = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.tree_components = tree_components
        # list indexed by ComponentType (int enum 0..N-1)
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
        self.last_access_time = get_and_increase_time_counter()
        self.hash_value = None
        self.hit_count = 0
        self.lru_prev: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.lru_next: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1

    def component(self, component_type: ComponentType) -> ComponentData:
        return self.component_data[component_type]

    @property
    def backuped(self) -> bool:
        """Tree-level: Full KV present on host."""
        return self.component_data[ComponentType.FULL].host_value is not None

    @property
    def evicted(self) -> bool:
        """Tree-level: Full KV not on device (non-root with value=None)."""
        return (
            self.parent is not None
            and self.component_data[ComponentType.FULL].value is None
        )

    def __lt__(self, other: UnifiedTreeNode):
        return self.last_access_time < other.last_access_time

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: "UnifiedTreeNode") -> list[str]:
        """Returns all hash values from root to node (inclusive)."""
        if node is None or node.hash_value is None:
            return []
        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def protect_host(self) -> None:
        """Protect the host value from eviction (for L3 storage prefetch)."""
        self.component_data[ComponentType.FULL].host_lock_ref += 1

    def release_host(self) -> None:
        """Release the host value, allowing it to be evicted."""
        cd = self.component_data[ComponentType.FULL]
        if cd.host_lock_ref > 0:
            cd.host_lock_ref -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")


class UnifiedLRUList:
    def __init__(
        self,
        component_type: ComponentType,
        tree_components: tuple[ComponentType, ...],
        use_host_ptr: bool = False,
    ):
        self.component_type = component_type
        self.use_host_ptr = use_host_ptr
        # Pointer slot: host LRU uses offset slots so device/host pointers
        # never collide on the same node.
        self._pt: int = component_type + (_NUM_COMPONENT_TYPES if use_host_ptr else 0)
        self.head = UnifiedTreeNode(tree_components)
        self.tail = UnifiedTreeNode(tree_components)
        self.head.lru_next[self._pt] = self.tail
        self.tail.lru_prev[self._pt] = self.head
        self.cache: dict[int, UnifiedTreeNode] = {}

    def _add_node_after(self, prev_node: UnifiedTreeNode, new_node: UnifiedTreeNode):
        pt = self._pt
        new_node.lru_prev[pt] = prev_node
        new_node.lru_next[pt] = prev_node.lru_next[pt]
        prev_node.lru_next[pt].lru_prev[pt] = new_node
        prev_node.lru_next[pt] = new_node

    def _add_node(self, node: UnifiedTreeNode):
        self._add_node_after(self.head, node)

    def _remove_node(self, node: UnifiedTreeNode):
        pt = self._pt
        node.lru_prev[pt].lru_next[pt] = node.lru_next[pt]
        node.lru_next[pt].lru_prev[pt] = node.lru_prev[pt]

    def insert_mru(self, node: UnifiedTreeNode):
        assert node.id not in self.cache
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        del self.cache[node.id]
        self._remove_node(node)

    def reset_node_mru(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        should_include,
    ):
        prev_node = self.head
        while node != root_node:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def in_list(self, node: Optional[UnifiedTreeNode]):
        return node is not None and node.id in self.cache

    def _locked(self, node: UnifiedTreeNode) -> bool:
        cd = node.component_data[self.component_type]
        return cd.host_lock_ref > 0 if self.use_host_ptr else cd.lock_ref > 0

    def get_prev_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        x = node.lru_prev[pt]
        while self._locked(x):
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        x = node.lru_prev[pt]
        while self._locked(x) or len(x.children) > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_lru_no_lock(self):
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self):
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
    ComponentType.SHARED_ANCHOR: SharedAnchorComponent,
}

logger = logging.getLogger(__name__)


class UnifiedRadixCache(BasePrefixCache):
    def __init__(
        self,
        params: CacheInitParams,
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.is_eagle = params.is_eagle

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        assert params.tree_components is not None
        self.tree_components = tuple(params.tree_components)
        self.components: dict[ComponentType, TreeComponent] = {
            ct: COMPONENT_REGISTRY[ct](self, params) for ct in self.tree_components
        }
        self._components_tuple: tuple[TreeComponent, ...] = tuple(
            self.components.values()
        )
        if self.is_eagle:
            self.key_convert_fn = convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key

        # Streaming session: embedded StreamingSession with self as inner.
        # Always on -- zero overhead when no streaming session is open (the
        # try_* entries short-circuit on non-streaming reqs / real TreeNodes).
        # Dispatch methods below pre-check conditions so the session's
        # internal fall-through to self.inner.xxx never fires -- no recursion.
        self.session = StreamingSession(inner=self)

        self.tp_group = params.tp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )

        # HiCache D↔H defaults (overridden by init_hicache)
        self.cache_controller = None
        self.write_through_threshold = 256

        self.reset()
        logger.info(f"Init Unified RadixTree with components {self.tree_components}")

    def reset(self) -> None:
        self._reset_full()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        extra_config = {}
        if storage_backend_extra_config:
            if storage_backend_extra_config.startswith("@"):
                path = storage_backend_extra_config[1:]
                ext = os.path.splitext(path)[1].lower()
                with open(path, "rb" if ext == ".toml" else "r") as f:
                    if ext == ".json":
                        extra_config = json.load(f)
                    elif ext == ".toml":
                        import tomllib

                        extra_config = tomllib.load(f)
                    elif ext in (".yaml", ".yml"):
                        import yaml

                        extra_config = yaml.safe_load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file {path} (config format: {ext})"
                        )
            else:
                extra_config = json.loads(storage_backend_extra_config)

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )
        return (
            extra_config,
            int(prefetch_threshold),
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            bool(hicache_storage_pass_prefix_keys),
        )

    def _reset_full(self) -> None:
        """Full reset: destroy entire tree and all state."""
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.root_node.key = RadixKey([], None)
        self.root_node.hash_value = []
        self.root_node.component_data[BASE_COMPONENT_TYPE].value = []
        for ct in self.tree_components:
            self.root_node.component_data[ct].lock_ref = 1
        self.component_evictable_size_ = {ct: 0 for ct in self.tree_components}
        self.component_protected_size_ = {ct: 0 for ct in self.tree_components}

        self.lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components) for ct in self.tree_components
        }
        self.session.slots.clear()

        self.evictable_device_leaves: set[UnifiedTreeNode] = set()
        self.evictable_host_leaves: set[UnifiedTreeNode] = set()
        self.host_lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components, use_host_ptr=True)
            for ct in self.tree_components
        }
        # Scheduler-thread owned D↔H bookkeeping. The controller records CUDA
        # stream work and exposes ack lists, but it does not mutate these maps.
        # L3 storage worker threads communicate through Queue instances instead.
        self.ongoing_write_through: dict[int, UnifiedTreeNode] = {}
        self.ongoing_load_back: dict[int, UnifiedTreeNode] = {}
        self.enable_storage = False
        self.hicache_storage_pass_prefix_keys = False
        self.prefetch_threshold = 256
        self.prefetch_stop_policy = "best_effort"
        self.prefetch_timeout_base = 3.0
        self.prefetch_timeout_per_page = 0.001
        self.ongoing_prefetch: dict = {}
        self.ongoing_backup: dict = {}
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}

        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()

    def init_hicache(self, server_args: ServerArgs, params: CacheInitParams) -> None:
        """Initialize HiCache infrastructure."""
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            attach_hybrid_pool_to_unified_cache,
        )

        # Direct IO layout fixup (must happen before pool creation)
        if server_args.hicache_io_backend == "direct":
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, "
                    "switching to page first direct layout"
                )

        self.load_cache_event = threading.Event()
        attach_hybrid_pool_to_unified_cache(
            self,
            params,
            server_args,
            load_cache_event=self.load_cache_event,
        )

        # State initialization
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 256

        # L3 Storage initialization
        self.enable_storage = server_args.hicache_storage_backend is not None
        if self.enable_storage:
            (
                extra_config,
                extra_prefetch_threshold,
                extra_timeout_base,
                extra_timeout,
                extra_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                server_args.hicache_storage_backend_extra_config
            )
            prefetch_threshold = extra_prefetch_threshold
            prefetch_timeout_per_ki_token = extra_timeout
            prefetch_timeout_per_page = (
                self.page_size / 1024 * prefetch_timeout_per_ki_token
            )
            self.cache_controller.attach_storage_backend(
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                host_pools=self.host_pool_group.entries,
            )
            self.prefetch_threshold = prefetch_threshold
            self.prefetch_timeout_base = extra_timeout_base
            self.prefetch_timeout_per_page = prefetch_timeout_per_page
            self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy
            self.hicache_storage_pass_prefix_keys = (
                getattr(server_args, "hicache_storage_pass_prefix_keys", False)
                or extra_pass_prefix_keys
            )

        logger.info(
            f"HiCache D\u2194H initialized: "
            f"host_pool_size={self.host_pool_group.size}, "
            f"write_policy={server_args.hicache_write_policy}, "
            f"tp_world_size={self.tp_world_size}, "
            f"transfer_layer_num={self.cache_controller.layer_num}, "
            f"enable_storage={self.enable_storage}"
        )

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )
        key = key.page_aligned(self.page_size)

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        result = self._insert_helper(self.root_node, key, value, params)
        return result

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        for component in self._components_tuple:
            component.drive_eviction(params=params, tracker=tracker)

        self.update_eviction_metrics(sum(tracker.values()), start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(node=node, result=result)

        self._update_evictable_leaf_sets(node)
        return result

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        result = self.session.try_dec_lock_ref(node, params)
        if result is not None:
            return result
        if self.disable:
            return DecLockRefResult()
        for component in self._components_tuple:
            component.release_component_lock(node=node, params=params)

        self._update_evictable_leaf_sets(node)
        # TODO: delta is not aggregated from components; no caller uses it yet.
        return DecLockRefResult()

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs) -> None:
        if self.session.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return

        kv_committed_len = req.pop_committed_kv_cache()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(req, is_finished=True)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        result = None
        insert_params = None

        if is_insert:
            insert_params = InsertParams(prev_prefix_len=req.cache_protected_len)

            # components prepare insert data + return effective cache_len
            effective_cache_len = len(token_ids)
            for comp in self._components_tuple:
                cl = comp.prepare_for_caching_req(
                    req=req,
                    insert_params=insert_params,
                    token_ids_len=len(token_ids),
                    is_finished=True,
                )
                if cl is not None:
                    effective_cache_len = min(effective_cache_len, cl)

            # Truncate if needed
            if effective_cache_len < len(token_ids):
                free_start = max(effective_cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[free_start:])
                token_ids = token_ids[:effective_cache_len]
                kv_indices = kv_indices[:effective_cache_len]

            radix_key = RadixKey(
                token_ids, req.extra_key, is_bigram=self.is_eagle
            ).page_aligned(self.page_size)
            page_aligned_len = len(radix_key)
            values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

            insert_params.key = radix_key
            insert_params.value = values
            result = self.insert(insert_params)

            # Free unaligned tail
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req, is_finished=True, insert_result=result, insert_params=insert_params
            )

    def cache_unfinished_req(self, req: Req, chunked=False, **kwargs) -> None:
        if self.session.try_cache_unfinished_req(req, chunked=chunked, **kwargs):
            return

        token_ids = req.fill_ids

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # components prepare insert data + return effective cache_len
        insert_params = InsertParams(
            prev_prefix_len=req.cache_protected_len, chunked=chunked
        )
        effective_cache_len = len(token_ids)
        for comp in self._components_tuple:
            cl = comp.prepare_for_caching_req(
                req=req,
                insert_params=insert_params,
                token_ids_len=len(token_ids),
                is_finished=False,
            )
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        if effective_cache_len <= 0:
            req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req, is_finished=False, insert_params=insert_params
                )
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        radix_key = RadixKey(
            token_ids[:effective_cache_len],
            req.extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

        if ComponentType.SWA in self.components and req.swa_evicted_seqlen > 0:
            valid_swa_suffix_len = page_aligned_len - req.swa_evicted_seqlen
            sliding_window_size = self.components[ComponentType.SWA].sliding_window_size
            if 0 < valid_swa_suffix_len < sliding_window_size:
                req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
                for comp in self._components_tuple:
                    comp.cleanup_after_caching_req(
                        req, is_finished=False, insert_params=insert_params
                    )
                return

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        # Match prefix
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        new_prefix_len = result.prefix_len
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        lock_result = self.inc_lock_ref(new_last_node)

        # Update req fields
        if len(new_indices) < len(kv_indices_orig):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices_orig[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.cache_protected_len = len(new_indices)
        req.last_node = new_last_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req,
                is_finished=False,
                insert_result=result,
                insert_params=insert_params,
            )

    # ---- Internal Helpers ----

    def _match_prefix_helper_readonly(
        self, key: RadixKey
    ) -> tuple[list[torch.Tensor], UnifiedTreeNode, int]:
        """Read-only version of _match_prefix_helper that does not split nodes.
        Only considers fully matched nodes, ignores partial matches.

        Not used yet; reserved for future read-only match operations."""
        node = self.root_node
        child_key = key.child_key(self.page_size)
        value: list[torch.Tensor] = []
        best_value_len = 0
        best_node = node
        validators = tuple(
            comp.create_match_validator() for comp in self._components_tuple
        )

        def _update_best_if_valid(node):
            nonlocal best_value_len, best_node
            if all(v(node) for v in validators):
                best_value_len = len(value)
                best_node = node

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # HiCache: dead node (evicted + not backuped) — stop traversal
            if child.evicted and not child.backuped:
                break

            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                # Read-only: do not split, ignore partial match and stop
                break

            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)
        return value, best_node, best_value_len

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> tuple[list[torch.Tensor], UnifiedTreeNode, int]:
        node = self.root_node
        child_key = key.child_key(self.page_size)
        value: list[torch.Tensor] = []
        best_value_len = 0
        best_node = node
        validators = tuple(
            comp.create_match_validator() for comp in self._components_tuple
        )

        def _update_best_if_valid(node):
            nonlocal best_value_len, best_node
            if all(v(node) for v in validators):
                best_value_len = len(value)
                best_node = node

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # HiCache: dead node (evicted + not backuped) — stop traversal
            if child.evicted and not child.backuped:
                break

            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                if child.evicted:
                    break
                node = self._split_node(child.key, child, prefix_len)
                value.append(node.component_data[BASE_COMPONENT_TYPE].value)
                _update_best_if_valid(node)
                break

            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)
        return value, best_node, best_value_len

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        last_node: UnifiedTreeNode,
        best_value_len: int,
    ) -> MatchResult:
        node_update = last_node
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue  # Full uses last_access_time, not LRU
            self.lru_lists[comp.component_type].reset_node_and_parents_mru(
                node_update, self.root_node, comp.node_has_component_data
            )

        cur_time = get_and_increase_time_counter()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        # Walk up to find last_device_node
        last_device_node = last_node
        while last_device_node is not self.root_node and last_device_node.evicted:
            last_device_node = last_device_node.parent

        # Walk up to find last_host_node
        last_host_node = last_node
        while last_host_node is not self.root_node and not last_host_node.backuped:
            last_host_node = last_host_node.parent

        if best_value_len > 0:
            device_indices = torch.cat(value[:best_value_len])
        else:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=last_device_node,
            last_host_node=last_host_node,
            host_hit_length=0,
        )

        for component in self._components_tuple:
            result = component.finalize_match_result(
                result=result,
                params=params,
                value_chunks=value,
                best_value_len=best_value_len,
            )
        return result

    def _split_node(
        self, key: RadixKey, child: UnifiedTreeNode, split_len: int
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        self._for_each_component_lru(child, UnifiedLRUList.remove_node)

        child.parent = new_node
        child.key = child.key[split_len:]

        for component in self._components_tuple:
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        self._for_each_component_lru(
            new_node, UnifiedLRUList.insert_mru, skip_existing=True
        )
        self._for_each_component_lru(
            child, UnifiedLRUList.insert_mru, skip_existing=True
        )
        child.last_access_time = get_and_increase_time_counter()

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(child)
        return new_node

    def _touch_node(self, node: UnifiedTreeNode):
        node.last_access_time = get_and_increase_time_counter()
        if node != self.root_node:
            self._for_each_component_lru(node, UnifiedLRUList.reset_node_mru)

    def _add_new_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components)
        new_node.parent = parent
        new_node.key = key
        new_node.component_data[BASE_COMPONENT_TYPE].value = value.clone()
        if self.enable_storage:
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)
        parent.children[key.child_key(self.page_size)] = new_node
        self.component_evictable_size_[BASE_COMPONENT_TYPE] += len(value)

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(parent)
        return new_node

    def _unevict_node_on_insert(
        self, node: UnifiedTreeNode, fresh_value: torch.Tensor
    ) -> None:
        """Restore an evicted node's Full device value from fresh KV indices
        during insert."""
        ct = BASE_COMPONENT_TYPE
        cd = node.component_data[ct]
        assert cd.value is None
        n = len(fresh_value)
        cd.value = fresh_value.clone()
        self.component_evictable_size_[ct] += n
        self._update_evictable_leaf_sets(node)
        if node.parent is not None:
            self._update_evictable_leaf_sets(node.parent)

    def _insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> InsertResult:
        self._touch_node(node)
        if len(key) == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = key.child_key(self.page_size)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            if node.evicted:
                self._unevict_node_on_insert(node, value[:prefix_len])
                # FULL was restored from the request's fresh KV. Aux
                # components (e.g. SWA) may still hold tombstones and need
                # to rebuild their value from the same slice.
                for component in self._components_tuple:
                    if component.component_type == BASE_COMPONENT_TYPE:
                        continue
                    component.recover_after_unevict(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        params=params,
                    )
            else:
                value_slice = value[:prefix_len]
                consumed_from = prefix_len
                # Let each component claim ownership of overlapping KV slots
                for component in self._components_tuple:
                    comp_consumed_from = component.update_component_on_insert_overlap(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        value_slice=value_slice,
                        params=params,
                    )
                    consumed_from = min(consumed_from, comp_consumed_from)

                dup_start = max(0, params.prev_prefix_len - total_prefix_length)
                if dup_start < consumed_from:
                    self.token_to_kv_pool_allocator.free(
                        value_slice[dup_start:consumed_from]
                    )

            self._inc_hit_count(node, params.chunked)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

        is_new_leaf = False
        # Create new leaf for remaining suffix
        if len(key):
            if any(
                comp.should_skip_leaf_creation(
                    total_prefix_len=total_prefix_length,
                    key_len=len(key),
                    params=params,
                )
                for comp in self._components_tuple
            ):
                # TODO: When leaf creation is skipped, We should release all component
                # resources here or propagate a flag so that
                # cleanup_after_caching_req can free them properly.
                self.token_to_kv_pool_allocator.free(value)
                return InsertResult(prefix_len=total_prefix_length)
            target_node = self._add_new_node(node, key, value)
            is_new_leaf = True
        else:
            target_node = node

        # Finalize: let each component attach its data to the target node.
        # e.g. Mamba attaches mamba_value to the leaf node
        result = InsertResult(prefix_len=total_prefix_length)
        for component in self._components_tuple:
            component.commit_insert_component_data(
                node=target_node,
                is_new_leaf=is_new_leaf,
                params=params,
                result=result,
            )
        if is_new_leaf:
            self._inc_hit_count(target_node, params.chunked)
        return result

    # ---- Evict Helpers ----

    def _cascade_evict(
        self,
        node: UnifiedTreeNode,
        trigger: TreeComponent,
        tracker: dict[ComponentType, int],
        target: EvictLayer = EvictLayer.DEVICE,
    ):
        """Cascade eviction from trigger to lower-or-equal priority components."""
        is_leaf = len(node.children) == 0
        trigger_priority = trigger.eviction_priority(is_leaf)

        for comp in self._components_tuple:
            if comp.eviction_priority(is_leaf) <= trigger_priority:
                if comp is not trigger and comp.node_has_component_data(node, target):
                    cd = node.component_data[comp.component_type]
                    if EvictLayer.DEVICE in target:
                        assert cd.lock_ref == 0
                    if EvictLayer.HOST in target:
                        assert cd.host_lock_ref == 0
                    self._evict_component_and_detach_lru(
                        node, comp, target=target, tracker=tracker
                    )

        # Now that all components (including SWA which depends on Full.value)
        # have been freed, we can safely tombstone Full.value.
        # This is deferred from evict_component because free_swa needs it.
        if (
            EvictLayer.DEVICE in target
            and trigger.component_type == BASE_COMPONENT_TYPE
        ):
            self._finalize_full_device_eviction(node)

        self._update_evictable_leaf_sets(node)

    def _finalize_full_device_eviction(self, node: UnifiedTreeNode) -> None:
        """Tombstone Full.value after all coupled auxiliary device data is freed."""
        cd = node.component_data[BASE_COMPONENT_TYPE]
        assert cd.value is not None, (
            "Full.value must stay present until auxiliary components complete "
            "device eviction because they may need the full indices."
        )
        cd.value = None
        assert cd.value is None

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode):
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node

    def _evict_component_and_detach_lru(
        self,
        node: UnifiedTreeNode,
        comp: TreeComponent,
        target: EvictLayer = EvictLayer.DEVICE,
        tracker: dict[ComponentType, int] = None,
    ) -> tuple[int, int]:
        device_freed, host_freed = comp.evict_component(node, target=target)
        if tracker is not None:
            if EvictLayer.DEVICE in target:
                tracker[comp.component_type] += device_freed
            elif EvictLayer.HOST in target:
                tracker[comp.component_type] += host_freed

        # Detach from the appropriate LRU list(s)
        ct = comp.component_type
        for layer, lru_lists in (
            (EvictLayer.DEVICE, self.lru_lists),
            (EvictLayer.HOST, self.host_lru_lists),
        ):
            if layer in target:
                lru = lru_lists[ct]
                if lru.in_list(node):
                    lru.remove_node(node)
        return device_freed, host_freed

    def _iteratively_delete_tombstone_leaf(
        self, deleted_node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ):
        """Walk up from *deleted_node* and cascade-delete childless ancestors.

        Only the Full (base) component decides whether a node survives:
          - Full device present  → keep as D-leaf
          - Full host present    → keep as H-leaf
          - neither              → evict all remaining data, delete, continue up
        """
        ct = BASE_COMPONENT_TYPE
        cur = deleted_node.parent
        while cur != self.root_node and len(cur.children) == 0:
            if any(
                cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in cur.component_data
            ):
                break

            has_device = cur.component_data[ct].value is not None
            has_host = cur.component_data[ct].host_value is not None

            if has_device:
                self._update_evictable_leaf_sets(cur)
                break

            # Full device absent — clean up orphaned aux device data.
            for comp in self.components.values():
                if comp.node_has_component_data(cur):
                    self._evict_component_and_detach_lru(
                        cur, comp, target=EvictLayer.DEVICE, tracker=tracker
                    )

            if has_host:
                self._update_evictable_leaf_sets(cur)
                break

            # Full absent on both layers — evict remaining host data, delete.
            for comp in self.components.values():
                if comp.node_has_component_data(cur, target=EvictLayer.HOST):
                    self._evict_component_and_detach_lru(
                        cur, comp, target=EvictLayer.HOST, tracker=tracker
                    )

            self.evictable_host_leaves.discard(cur)
            self._remove_leaf_from_parent(cur)
            parent = cur.parent
            self._update_evictable_leaf_sets(parent)
            cur = parent

    def _for_each_component_lru(
        self,
        node: UnifiedTreeNode,
        lru_op,
        target: EvictLayer = EvictLayer.DEVICE,
        skip_existing: bool = False,
    ):
        """Apply lru_op to each aux component's LRU that has data on this node.
        If skip_existing=True, skip components already in the target LRU list."""
        lru_dict = self.host_lru_lists if target is EvictLayer.HOST else self.lru_lists
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue  # Full uses leaf sets, not LRU
            cd = node.component_data[ct]
            if (cd.host_value if target is EvictLayer.HOST else cd.value) is not None:
                lru = lru_dict[ct]
                if skip_existing and lru.in_list(node):
                    continue
                lru_op(lru, node)

    def evict_host(
        self, num_tokens: int, component_type: ComponentType = BASE_COMPONENT_TYPE
    ) -> int:
        """Evict host resources for a specific component to free host pool space."""
        tracker: dict[ComponentType, int] = {ct: 0 for ct in self.tree_components}
        comp = self.components.get(component_type)
        if comp is not None:
            comp.drive_host_eviction(num_tokens, tracker)
        return tracker[component_type]

    def _is_device_leaf(self, node: UnifiedTreeNode) -> bool:
        """D-leaf: Full device value present, no child with Full KV on device,
        unlocked, not root.

        Only the Full (base) component is required; auxiliary components
        (Mamba, SWA) are not mandatory for D-leaf membership."""
        ct = BASE_COMPONENT_TYPE
        if node is self.root_node or node.evicted:
            return False
        if any(cd.lock_ref > 0 for cd in node.component_data):
            return False
        if any(
            child.component_data[ct].value is not None
            for child in node.children.values()
        ):
            return False
        return True

    def _is_host_leaf(self, node: UnifiedTreeNode) -> bool:
        """H-leaf: evicted, Full host value present, no children, unlocked, not root.

        Only the Full (base) component host_value is required; auxiliary
        components are not mandatory for H-leaf membership."""
        if node is self.root_node or not node.evicted:
            return False
        if not node.backuped:
            return False
        if any(cd.host_lock_ref > 0 for cd in node.component_data):
            return False
        if len(node.children) > 0:
            return False
        return True

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        """Update both device and host leaf sets for a node."""
        if self._is_device_leaf(node):
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)

        if self._is_host_leaf(node):
            self.evictable_host_leaves.add(node)
        else:
            self.evictable_host_leaves.discard(node)

    def _evict_to_host(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int] = None
    ) -> None:
        """GPU→CPU demotion: release all device resources, node stays in tree."""
        assert not node.evicted and node.backuped
        trigger = self.components[BASE_COMPONENT_TYPE]
        self._evict_component_and_detach_lru(
            node, trigger, target=EvictLayer.DEVICE, tracker=tracker
        )
        self._cascade_evict(node, trigger, tracker)

        # after device eviction, insert aux components into host LRU.
        self._for_each_component_lru(
            node, UnifiedLRUList.insert_mru, target=EvictLayer.HOST, skip_existing=True
        )
        self._update_evictable_leaf_sets(node.parent)

    def _evict_device_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict a device leaf node, choosing the right strategy:

        - backuped: demote to host via _evict_to_host (node stays in tree)
        - not backuped + write_back: write_backup first, then demote
        - not backuped + write_through: Cascade evict all components

        All freed device tokens are accumulated into *tracker*.
        """
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        if not node.backuped:
            if (
                self.cache_controller is not None
                and self.cache_controller.write_policy == "write_back"
            ):
                self.write_backup(node, write_back=True)
                self._evict_to_host(node, tracker)
                return
            else:
                # Write-through: node has no backup, delete entirely.
                for comp in self._components_tuple:
                    self._evict_component_and_detach_lru(
                        node, comp, target=EvictLayer.ALL, tracker=tracker
                    )
                self._finalize_full_device_eviction(node)
                self.evictable_device_leaves.discard(node)
                parent = node.parent
                self._remove_leaf_from_parent(node)
                self._update_evictable_leaf_sets(parent)
                self._iteratively_delete_tombstone_leaf(node, tracker)
                return
        self._evict_to_host(node, tracker)

    def _evict_host_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> None:
        """Atomically evict all components on a host leaf.

        All freed tokens are accumulated into *tracker*."""
        assert self._is_host_leaf(node), f"node {node.id} is not an H-leaf"

        for comp in self._components_tuple:
            _, hf = self._evict_component_and_detach_lru(
                node, comp, target=EvictLayer.ALL, tracker=None
            )
            tracker[comp.component_type] += hf
        self.evictable_host_leaves.discard(node)
        self._remove_leaf_from_parent(node)
        self._iteratively_delete_tombstone_leaf(node, tracker)

    # ---- HiCache: Backup / LoadBack ----

    def _ensure_hash_values(self, node: UnifiedTreeNode) -> None:
        if node is None:
            return
        if node.parent is not None:
            self._ensure_hash_values(node.parent)
        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)

    def _protect_host_node(self, node: UnifiedTreeNode, protect_aux: bool = True) -> None:
        node.protect_host()
        self.evictable_host_leaves.discard(node)
        if not protect_aux:
            return
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            cd = node.component_data[ct]
            if cd.host_value is None:
                continue
            if self.host_lru_lists[ct].in_list(node):
                self.host_lru_lists[ct].remove_node(node)
            cd.host_lock_ref += 1

    def _release_host_node(
        self, node: UnifiedTreeNode, release_aux: bool = True
    ) -> None:
        node.release_host()
        if release_aux:
            for ct in self.tree_components:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                cd = node.component_data[ct]
                if cd.host_lock_ref == 0:
                    continue
                cd.host_lock_ref -= 1
                if (
                    cd.host_lock_ref == 0
                    and cd.value is None
                    and cd.host_value is not None
                    and not self.host_lru_lists[ct].in_list(node)
                ):
                    self.host_lru_lists[ct].insert_mru(node)
        self._update_evictable_leaf_sets(node)

    def _swa_storage_transfers(
        self, node: UnifiedTreeNode
    ) -> Optional[list[PoolTransfer]]:
        if ComponentType.SWA not in self.components:
            return None
        cd = node.component_data[ComponentType.SWA]
        if cd.host_value is None:
            return None
        self._ensure_hash_values(node)
        if not node.hash_value:
            return None
        num_pages = len(cd.host_value) // self.page_size
        if num_pages <= 0:
            return None
        return [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=cd.host_value,
                keys=node.hash_value[-num_pages:],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def _alloc_swa_prefetch_transfers(
        self, prefetch_length: int
    ) -> Optional[list[PoolTransfer]]:
        if ComponentType.SWA not in self.components:
            return None
        if self.swa_kv_pool_host is None:
            return None

        sliding_window_size = self.components[ComponentType.SWA].sliding_window_size
        num_swa_pages = min(
            prefetch_length // self.page_size,
            (sliding_window_size + self.page_size - 1) // self.page_size,
        )
        if num_swa_pages <= 0:
            return None
        num_swa_tokens = num_swa_pages * self.page_size
        host_indices = self.swa_kv_pool_host.alloc(num_swa_tokens)
        if host_indices is None:
            self.evict_host(num_swa_tokens, ComponentType.SWA)
            host_indices = self.swa_kv_pool_host.alloc(num_swa_tokens)
        if host_indices is None:
            return None
        return [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=host_indices,
                keys=["__placeholder__"] * num_swa_pages,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def _free_prefetch_extra_pools(
        self, transfers: Optional[list[PoolTransfer]]
    ) -> None:
        for transfer in transfers or []:
            if transfer.host_indices is None:
                continue
            if transfer.name == PoolName.SWA and self.swa_kv_pool_host is not None:
                self.swa_kv_pool_host.free(transfer.host_indices)

    def write_backup_storage(self, node: UnifiedTreeNode) -> None:
        if (
            not self.enable_storage
            or node.component_data[BASE_COMPONENT_TYPE].host_value is None
        ):
            return
        self._ensure_hash_values(node)
        if not node.hash_value:
            return
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )
        extra_pools = self._swa_storage_transfers(node)
        operation_id = self.cache_controller.write_storage(
            node.component_data[BASE_COMPONENT_TYPE].host_value,
            node.key,
            node.hash_value,
            prefix_keys,
            extra_pools=extra_pools,
        )
        self.ongoing_backup[operation_id] = node
        self._protect_host_node(node, protect_aux=extra_pools is not None)

    def write_backup(self, node: UnifiedTreeNode, write_back: bool = False) -> int:
        """Backup a node's data from device to host (D->H)."""
        if self.cache_controller is None:
            return 0

        # Backup invariant (write-through): parent must be backuped first
        if not write_back and (
            node.parent is not self.root_node and not node.parent.backuped
        ):
            return 0

        # Build aux transfers, keyed per component
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_HOST)
            if t:
                comp_xfers[comp.component_type] = t

        # Pre-evict host if insufficient
        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        kv_tokens = len(device_value)
        host_avail = self.cache_controller.mem_pool_host.available_size()
        if host_avail < kv_tokens:
            needed = kv_tokens - host_avail
            evicted = self.evict_host(needed)
            if evicted < needed:
                return 0

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        host_indices = self.cache_controller.write(
            device_value, node_id=node.id, extra_pools=aux_xfers or None
        )
        if host_indices is None:
            return 0

        # Commit
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[kv_xfer],
        )
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                transfers=xfers,
            )

        self.ongoing_write_through[node.id] = node
        if not write_back:
            self.inc_lock_ref(node)
        return len(host_indices)

    def load_back(
        self,
        node: UnifiedTreeNode,
        mem_quota: Optional[int] = None,
        req=None,
    ) -> Optional[torch.Tensor]:
        """Load evicted KV data from host back to device (H→D)."""
        if self.cache_controller is None:
            return None

        # Build KV transfer
        last_hit_node = node
        kv_xfer = self.components[BASE_COMPONENT_TYPE].build_hicache_transfers(
            last_hit_node, CacheTransferPhase.LOAD_BACK
        )[0]

        # Lock path & pre-evict if device pool is insufficient
        nodes_to_load = kv_xfer.nodes_to_load
        ancestor_node = nodes_to_load[0].parent if nodes_to_load else last_hit_node
        result = self.inc_lock_ref(ancestor_node)
        kv_tokens = len(kv_xfer.host_indices)

        # Build aux transfers, keyed per component.
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(
                last_hit_node,
                CacheTransferPhase.LOAD_BACK,
                req=req,
                max_suffix_tokens=kv_tokens,
            )
            if t:
                comp_xfers[comp.component_type] = t

        # Aux builders, especially SWA, may split host-only nodes to cap the
        # transferred suffix. Rebuild the Full-KV transfer after those splits so
        # nodes_to_load matches the final path that inc_lock_ref will lock.
        kv_xfer = self.components[BASE_COMPONENT_TYPE].build_hicache_transfers(
            last_hit_node, CacheTransferPhase.LOAD_BACK
        )[0]
        nodes_to_load = kv_xfer.nodes_to_load
        kv_tokens = len(kv_xfer.host_indices)

        # Skip if there is nothing to load, or if the Full-KV transfer is too
        # small / exceeds memory quota. Aux transfers should still run even
        # when the Full-KV load is skipped by thresholding.
        if (kv_tokens < self.load_back_threshold and not comp_xfers) or (
            mem_quota is not None and kv_tokens > mem_quota + result.delta
        ):
            self.dec_lock_ref(ancestor_node)
            return None

        full_avail = getattr(
            self.token_to_kv_pool_allocator,
            "full_available_size",
            self.token_to_kv_pool_allocator.available_size,
        )()
        swa_needed = 0
        if ComponentType.SWA in comp_xfers:
            swa_needed = comp_xfers[ComponentType.SWA][0].swa_suffix_tokens
        swa_avail = (
            self.token_to_kv_pool_allocator.swa_available_size()
            if swa_needed
            else swa_needed
        )
        full_shortage = max(0, kv_tokens - full_avail)
        swa_shortage = max(0, swa_needed - swa_avail)
        needed = max(full_shortage, swa_shortage)
        if needed > 0:
            result = self.evict(
                EvictParams(num_tokens=full_shortage, swa_num_tokens=swa_shortage)
            )
            if (
                result.num_tokens_evicted < full_shortage
                or result.swa_num_tokens_evicted < swa_shortage
            ):
                self.dec_lock_ref(ancestor_node)
                return None

        logger.info(
            "load_back: kv_tokens=%d, node_id=%d",
            kv_tokens,
            last_hit_node.id,
        )

        # Load H→D
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        for xfer in aux_xfers:
            if xfer.name == PoolName.SWA and xfer.swa_suffix_tokens > kv_tokens:
                xfer.swa_suffix_tokens = kv_tokens
                xfer.host_indices = xfer.host_indices[-kv_tokens:]
        device_indices = self.cache_controller.load(
            host_indices=kv_xfer.host_indices,
            node_id=last_hit_node.id,
            extra_pools=aux_xfers or None,
        )

        self.dec_lock_ref(ancestor_node)
        if device_indices is None:
            return None

        # Commit: each component gets only its own transfers
        kv_xfer.device_indices = device_indices
        self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            last_hit_node,
            CacheTransferPhase.LOAD_BACK,
            [kv_xfer],
        )
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                last_hit_node,
                CacheTransferPhase.LOAD_BACK,
                xfers,
            )

        self._update_evictable_leaf_sets(ancestor_node)
        self.inc_lock_ref(last_hit_node)
        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        return device_indices

    def _inc_hit_count(self, node: UnifiedTreeNode, chunked: bool = False) -> None:
        """Increment hit count; trigger write_backup when threshold reached."""
        if self.cache_controller is None:
            return
        if node.evicted or chunked:
            return
        if self.cache_controller.write_policy == "write_back":
            return
        node.hit_count += 1
        if not node.backuped and node.hit_count >= self.write_through_threshold:
            self.write_backup(node)

    # ---- HiCache: Async Event Management ----

    def writing_check(self, write_back: bool = False) -> None:
        """Poll write-through completions."""
        cc = self.cache_controller
        if cc is None:
            return

        if write_back:
            # Blocking: wait for all pending write-backs
            while self.ongoing_write_through:
                for _, finish_event, ack_list in cc.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        node = self.ongoing_write_through.pop(ack_id, None)
                        if self.enable_storage and node is not None:
                            self.write_backup_storage(node)
                cc.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in cc.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1

        # TP sync: MIN across all ranks for consistent tree updates
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                queue_size, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        finish_count = int(queue_size.item())

        # Process completed acks
        while finish_count > 0:
            _, finish_event, ack_list = cc.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                node = self.ongoing_write_through.pop(ack_id)
                self.dec_lock_ref(node)
                if self.enable_storage:
                    self.write_backup_storage(node)
            finish_count -= 1

    def loading_check(self) -> None:
        """Poll load-back completions."""
        cc = self.cache_controller
        if cc is None or not self.ongoing_load_back:
            return
        finish_count = 0
        for _, finish_event, ack_list in cc.ack_load_queue:
            if not finish_event.query():
                break
            finish_count += 1
            for ack_id in ack_list:
                node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(node)
        del cc.ack_load_queue[:finish_count]

    # ---- HiCache: Scheduler Entry Points ----

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, UnifiedTreeNode]:
        """Prepare KV cache loading from host to device.
        Returns (device_indices, last_node) tuple."""
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        req = params.req

        if last_node.evicted or params.host_hit_length > 0:
            logger.info(
                "init_load_back triggered: node_id=%d, host_hit_length=%d",
                last_node.id,
                params.host_hit_length,
            )
            loading_values = self.load_back(last_node, mem_quota, req=req)
            if loading_values is not None:
                logger.info(
                    "init_load_back success: loaded %d tokens for node %d",
                    len(loading_values),
                    last_node.id,
                )
                return loading_values, last_node

            # Fallback: walk up to non-evicted ancestor
            while last_node is not self.root_node and last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def check_hicache_events(self) -> None:
        """Called per scheduler step to poll async HiCache events."""
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
    ) -> None:
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
            info = self.ongoing_prefetch.pop(req_id, None)
            if info is not None:
                last_host_node, token_ids, _, operation = info
                self._free_prefetch_extra_pools(operation.pool_transfers)
                self._release_host_node(last_host_node, release_aux=False)
                cc.prefetch_tokens_occupied -= len(token_ids)
                if cc.prefetch_tokens_occupied < 0:
                    cc.prefetch_tokens_occupied = 0

        for operation in _drain_queue(cc.ack_backup_queue, n_backup):
            node = self.ongoing_backup.pop(operation.id, None)
            if node is not None:
                self._release_host_node(node)

        host_indices_list = list(_drain_queue(cc.host_mem_release_queue, n_release))
        if host_indices_list:
            cc.mem_pool_host.free(torch.cat(host_indices_list))

    def _drain_storage_control_queues_local(self) -> None:
        self._drain_storage_control_queues_impl(
            n_revoke=None, n_backup=None, n_release=None
        )

    def drain_storage_control_queues(self) -> None:
        cc = self.cache_controller
        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(n_revoke, n_backup, n_release)

    def flush_write_through_acks(self) -> None:
        """Flush pending write-through acknowledgements."""
        self.writing_check()

    def ready_to_load_host_cache(self) -> int:
        """Notify the cache controller to start the KV cache loading."""
        if self.cache_controller is not None:
            return self.cache_controller.start_loading()
        return 0

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: UnifiedTreeNode,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
    ) -> None:
        """Prefetch KV cache from L3 storage layer.

        This is called by the scheduler to initiate prefetching of KV cache
        data from persistent storage (L3) to host memory (L2).
        """
        new_input_tokens = (
            convert_to_bigram_key(new_input_tokens)
            if self.is_eagle
            else new_input_tokens
        )
        # Align the number of fetching tokens to the page size
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]

        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller is None
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        self._protect_host_node(last_host_node, protect_aux=False)
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self._release_host_node(last_host_node, release_aux=False)
            # No sufficient host memory for prefetch
            return

        extra_pools = self._alloc_swa_prefetch_transfers(prefetch_length)
        if ComponentType.SWA in self.components and extra_pools is None:
            self.cache_controller.mem_pool_host.free(host_indices)
            self._release_host_node(last_host_node, release_aux=False)
            return

        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys,
            extra_pools=extra_pools,
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)

    def can_terminate_prefetch(self, operation) -> bool:
        """Check if a prefetch operation can be terminated."""
        if self.prefetch_stop_policy == "best_effort":
            return True

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # Unknown prefetch stop policy, just return True
            return True

        operation_terminated = operation.is_terminated()
        if self.tp_world_size > 1:
            states = torch.tensor(
                [1 - int(can_terminate), int(operation_terminated)],
                dtype=torch.int,
            )
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )
            can_terminate = states[0].item() == 0
            operation_terminated = states[1].item() == 1

        if operation_terminated and not can_terminate:
            can_terminate = True
        return can_terminate

    def is_prefetch_timeout(self, operation) -> bool:
        """Check if a prefetch operation has timed out."""
        num_pages = len(operation.token_ids) // self.page_size
        timeout = self.prefetch_timeout_base + num_pages * self.prefetch_timeout_per_page
        return time.monotonic() - operation.start_time > timeout

    def check_prefetch_progress(self, req_id: str) -> bool:
        """Check if prefetch for a request is complete.

        Returns True if there is no ongoing prefetch or if prefetch is done.
        """
        if req_id not in self.ongoing_prefetch:
            return True

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            # Prefetch has not been issued due to insufficient host memory
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            completed_tokens_tensor = torch.tensor(
                min_completed_tokens, dtype=torch.int
            )
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_completed_tokens = completed_tokens_tensor.item()

        fetched_token_ids = token_ids[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        swa_host_indices = None
        swa_loaded_tokens = 0
        for transfer in operation.pool_transfers or []:
            if transfer.name == PoolName.SWA:
                swa_host_indices = transfer.host_indices
                swa_loaded_pages = operation.pool_storage_result.extra_pool_hit_pages.get(
                    PoolName.SWA, 0
                )
                if transfer.keys is not None:
                    swa_loaded_pages = min(swa_loaded_pages, len(transfer.keys))
                swa_loaded_tokens = swa_loaded_pages * self.page_size
                if swa_host_indices is not None:
                    swa_host_indices = swa_host_indices[:swa_loaded_tokens]
                break

        matched_length = self._insert_helper_host(
            last_host_node,
            RadixKey(
                token_ids=fetched_token_ids, extra_key=last_host_node.key.extra_key
            ),
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
            swa_host_indices=swa_host_indices,
            swa_loaded_tokens=swa_loaded_tokens,
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        if swa_host_indices is not None:
            inserted_new = matched_length < min_completed_tokens
            if not inserted_new or swa_loaded_tokens == 0:
                self.swa_kv_pool_host.free(swa_host_indices)
        self._release_host_node(last_host_node, release_aux=False)
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

        # Track tokens actually loaded from storage for this request (L3 hits)
        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        return True

    def _insert_helper_host(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
        swa_host_indices: Optional[torch.Tensor] = None,
        swa_loaded_tokens: int = 0,
    ) -> int:
        """Insert prefetched data from storage into the tree (host layer only)."""
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)
        matched_length = 0

        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            # Create new node with host-only data
            new_node = UnifiedTreeNode(self.tree_components)
            new_node.parent = node
            new_node.key = key
            # Device value is None (evicted), host value is present
            new_node.component_data[BASE_COMPONENT_TYPE].value = None
            new_node.component_data[BASE_COMPONENT_TYPE].host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[key.child_key(self.page_size)] = new_node
            self._update_evictable_leaf_sets(new_node)
            self._update_evictable_leaf_sets(node)

            if (
                ComponentType.SWA in self.components
                and swa_host_indices is not None
                and swa_loaded_tokens > 0
            ):
                target = new_node
                swa_loaded_tokens = min(swa_loaded_tokens, len(new_node.key))
                swa_host_indices = swa_host_indices[-swa_loaded_tokens:]
                if swa_loaded_tokens < len(new_node.key):
                    split_len = len(new_node.key) - swa_loaded_tokens
                    parent = self._split_node(new_node.key, new_node, split_len)
                    target = next(iter(parent.children.values()))

                cd = target.component_data[ComponentType.SWA]
                cd.host_value = swa_host_indices[:swa_loaded_tokens].clone()
                if not self.host_lru_lists[ComponentType.SWA].in_list(target):
                    self.host_lru_lists[ComponentType.SWA].insert_mru(target)

        return matched_length

    def terminate_prefetch(self, req_id: str) -> None:
        """Terminate an ongoing prefetch operation for a request."""
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """Pop and return the number of tokens loaded by prefetch.

        Returns 0 if no prefetch was done for this request.
        """
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def clear_storage_backend(self) -> bool:
        """Clear the storage backend state."""
        self.ongoing_prefetch.clear()
        self.ongoing_backup.clear()
        self.prefetch_loaded_tokens_by_reqid.clear()
        if self.enable_storage and hasattr(
            self.cache_controller.storage_backend, "clear"
        ):
            self.cache_controller.storage_backend.clear()
            return True
        return False

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach a storage backend for L3 caching.

        Returns (success, message) tuple.
        """
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    "Invalid hicache_storage_prefetch_policy: "
                    f"{hicache_storage_prefetch_policy!r}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}.",
                )

        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type
            if current_backend != storage_backend:
                return (
                    False,
                    f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                    f"Cannot attach different backend '{storage_backend}'. Detach first.",
                )
            if hicache_storage_prefetch_policy is not None:
                self.prefetch_stop_policy = hicache_storage_prefetch_policy
            if hicache_write_policy is not None:
                self.cache_controller.write_policy = hicache_write_policy
                self.write_through_threshold = (
                    1 if hicache_write_policy == "write_through" else 2
                )
            return True, "HiCache storage backend already enabled; policies updated."

        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            return False, f"Failed to parse storage backend extra config: {e}"

        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                host_pools=self.host_pool_group.entries,
            )
        except Exception as e:
            logger.exception("Failed to attach storage backend '%s'", storage_backend)
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self.enable_storage = True
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )
        self.hicache_storage_pass_prefix_keys = pass_prefix_keys
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach the current storage backend.

        Returns (success, message) tuple.
        """
        if not self.enable_storage:
            return True, "Storage backend already detached"
        try:
            self._drain_storage_control_queues_local()
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend")
            return False, f"Failed to detach storage backend: {e}"
        self.ongoing_prefetch.clear()
        self.ongoing_backup.clear()
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.enable_storage = False
        return True, "Storage backend detached"

    def release_aborted_request(self, rid: str) -> None:
        """Clean up storage prefetch state for an aborted request."""
        # Clean up storage hit tracking for aborted request
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            del self.ongoing_prefetch[rid]
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)
        self._release_host_node(last_host_node, release_aux=False)
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self._free_prefetch_extra_pools(operation.pool_transfers)
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

    # ---- Query / Inspection APIs ----
    # These APIs exist for compatibility with other RadixTree implementations.
    # TODO: simplify and consolidate in a future refactor.

    @property
    def sliding_window_size(self):
        swa = self.components.get(ComponentType.SWA)
        return swa.sliding_window_size if swa else None

    def supports_swa(self) -> bool:
        return ComponentType.SWA in self.components

    def supports_mamba(self) -> bool:
        return ComponentType.MAMBA in self.components

    # ---- Streaming session API (delegates to composed StreamingSession) ----

    def supports_streaming_session(self) -> bool:
        return True

    def release_session(self, session_id: str) -> None:
        self.session.release_session(session_id)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_tokens(active_pool_idxs)

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_full_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_swa_tokens(active_pool_idxs)

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_req_count(active_pool_idxs)

    def evictable_size(self) -> int:
        return self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)

    def protected_size(self) -> int:
        return self.component_protected_size_.get(BASE_COMPONENT_TYPE, 0)

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.SWA, 0)

    def mamba_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.MAMBA, 0)

    def swa_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.SWA, 0)

    def mamba_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.MAMBA, 0)

    def total_size(self):
        total_size = 0
        total_aux_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            if full_value is not None:
                total_size += len(full_value)
            for ct in self.tree_components:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                value = node.component_data[ct].value
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: UnifiedTreeNode):
            for child in node.children.values():
                v = child.component_data[BASE_COMPONENT_TYPE].value
                if v is not None:
                    values.append(v)
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def _all_component_values_flatten(
        self, component_type: ComponentType
    ) -> torch.Tensor:
        if component_type not in self.components:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        values = []

        def _dfs(node: UnifiedTreeNode):
            value = node.component_data[component_type].value
            if value is not None:
                values.append(value)
            for child in node.children.values():
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.MAMBA)

    def all_swa_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.SWA)

    def available_and_evictable_str(self) -> str:
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.component_evictable_size_[BASE_COMPONENT_TYPE]
        lines = [
            f"Available full tokens: {full_available_size + full_evictable} "
            f"(full_available_size={full_available_size} + full_evictable_size_={full_evictable})"
        ]
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            if ct.is_swa:
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif ct.is_mamba:
                available_size = self.req_to_token_pool.mamba_pool.available_size()
            else:
                continue

            lines.append(
                f"Available {ct}: {available_size + self.component_evictable_size_[ct]} "
                f"(available_size={available_size} + component_evictable_size_={self.component_evictable_size_[ct]})"
            )
        return "\n".join(lines) + "\n"

    def _collect_all_nodes(self) -> list[UnifiedTreeNode]:
        nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def sanity_check(self):
        """Verify tree invariants.

        TODO(hzh): This method has relatively high latency; simplify the
        check logic once the tree implementation stabilizes.
        """
        # Skip when streaming sessions hold tree locks: the check asserts
        # all nodes are unlocked during idle, which streaming sessions break
        # by design (they hold a first-turn lock across turns).
        if self.session.any_holding_kv():
            return

        errors: list[str] = []
        E = errors.append
        all_nodes = self._collect_all_nodes()
        all_node_set = set(all_nodes)
        FCT = BASE_COMPONENT_TYPE

        # ── PART 1: Tree Structure ──
        # Root state
        if self.root_node.component_data[FCT].value is None:
            E("[Root] root missing Full device value")
        if self.root_node.component_data[FCT].lock_ref <= 0:
            E(
                f"[Root] root Full lock_ref={self.root_node.component_data[FCT].lock_ref}"
            )
        if self.root_node.parent is not None:
            E("[Root] root has a parent pointer")
        # Parent ↔ child bidirectional consistency
        for node in all_nodes:
            for child in node.children.values():
                if child.parent is not node:
                    pid = child.parent.id if child.parent else None
                    E(f"[Tree] child {child.id} parent={pid}, expected {node.id}")
                if child.key is None:
                    E(f"[Tree] node {child.id} has no key")

        # ── PART 2: Per-Node State Machine (A2-A5) + Leaf Qualification ──
        expected_dev_leaves: set[UnifiedTreeNode] = set()
        expected_hst_leaves: set[UnifiedTreeNode] = set()

        for node in all_nodes:
            if node is self.root_node:
                continue
            nid = node.id
            full_dev = node.component_data[FCT].value is not None
            full_hst = node.component_data[FCT].host_value is not None

            # A2: Full is tree backbone — aux data requires Full data
            for ct in self.tree_components:
                if ct == FCT:
                    continue
                cd = node.component_data[ct]
                if cd.value is not None and not full_dev:
                    E(f"[A2] node {nid} {ct} device present but Full.value=None")
                if cd.host_value is not None and not full_hst:
                    E(f"[A2] node {nid} {ct} host present but Full.host_value=None")

            # A3: No dead nodes — at least Full device or Full host
            if not full_dev and not full_hst:
                E(f"[A3] node {nid} dead: no Full device and no Full host")

            # A4: Prefix continuity (parent must have data if child has)
            if node.parent is not None and node.parent is not self.root_node:
                p_dev = node.parent.component_data[FCT].value is not None
                p_hst = node.parent.component_data[FCT].host_value is not None
                if full_dev and not p_dev:
                    E(
                        f"[A4] node {nid} device present but parent {node.parent.id} evicted"
                    )
                if full_hst and not p_hst:
                    E(
                        f"[A4] node {nid} backed up but parent {node.parent.id} not backed up"
                    )

            # A5: Lock hierarchy + sanity
            fl = node.component_data[FCT].lock_ref
            for ct in self.tree_components:
                cd = node.component_data[ct]
                if cd.lock_ref < 0:
                    E(f"[A5] node {nid} {ct} lock_ref={cd.lock_ref}")
                if cd.host_lock_ref < 0:
                    E(f"[A5] node {nid} {ct} host_lock_ref={cd.host_lock_ref}")
                if ct != FCT and fl < cd.lock_ref:
                    E(f"[A5] node {nid} full_lock={fl} < {ct}_lock={cd.lock_ref}")
                if cd.value is None and cd.lock_ref > 0:
                    E(f"[A5] node {nid} {ct} evicted but lock_ref={cd.lock_ref}")

            # Collect expected leaf qualification (single pass)
            if self._is_device_leaf(node):
                expected_dev_leaves.add(node)
            if self._is_host_leaf(node):
                expected_hst_leaves.add(node)

        # ── PART 3: Tracking Structures (INV-1~5) ──

        # INV-3: D-leaf set matches expected
        if self.evictable_device_leaves != expected_dev_leaves:
            extra = self.evictable_device_leaves - expected_dev_leaves
            missing = expected_dev_leaves - self.evictable_device_leaves
            if extra:
                E(f"[INV-3] D-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"[INV-3] D-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # INV-4: H-leaf set matches expected
        if self.evictable_host_leaves != expected_hst_leaves:
            extra = self.evictable_host_leaves - expected_hst_leaves
            missing = expected_hst_leaves - self.evictable_host_leaves
            if extra:
                E(f"[INV-4] H-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"[INV-4] H-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # D-leaf ∩ H-leaf = ∅
        overlap = self.evictable_device_leaves & self.evictable_host_leaves
        if overlap:
            E(
                f"[Leaf] {len(overlap)} in both sets: {[n.id for n in list(overlap)[:5]]}"
            )

        # Stale nodes: leaf sets must only contain tree-reachable nodes
        stale = self.evictable_device_leaves - all_node_set
        if stale:
            E(
                f"[INV-3] {len(stale)} stale nodes in device_leaves: {[n.id for n in list(stale)[:5]]}"
            )
        stale = self.evictable_host_leaves - all_node_set
        if stale:
            E(
                f"[INV-4] {len(stale)} stale nodes in host_leaves: {[n.id for n in list(stale)[:5]]}"
            )

        # Per-component LRU tracking
        for ct in self.tree_components:
            lru = self.lru_lists[ct]
            if ct == FCT:
                # Full uses leaf sets, not LRU
                if len(lru.cache) > 0:
                    E(f"[INV-1] Full device LRU not empty: {len(lru.cache)}")
                if len(self.host_lru_lists[ct].cache) > 0:
                    E(
                        f"[INV-2] Full host LRU not empty: {len(self.host_lru_lists[ct].cache)}"
                    )
            else:
                # INV-1: Aux device value ↔ device LRU
                tree_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is not None
                }
                lru_ids = set(lru.cache.keys())
                if tree_ids != lru_ids:
                    E(
                        f"[INV-1] {ct} device LRU: "
                        f"+tree={tree_ids - lru_ids}, +lru={lru_ids - tree_ids}"
                    )
                # INV-2: Aux S3 (value=None, host_value!=None) ↔ host LRU
                host_lru = self.host_lru_lists[ct]
                s3_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is None
                    and n.component_data[ct].host_value is not None
                }
                host_lru_ids = set(host_lru.cache.keys())
                if s3_ids != host_lru_ids:
                    E(
                        f"[INV-2] {ct} host LRU: "
                        f"+S3={s3_ids - host_lru_ids}, +lru={host_lru_ids - s3_ids}"
                    )
                # INV-5: same aux not in both device and host LRU
                inv5_overlap = lru_ids & host_lru_ids
                if inv5_overlap:
                    E(f"[INV-5] {ct} in both device and host LRU: {inv5_overlap}")
                # Linked-list integrity
                self._check_lru_linked_list(lru, ct, "device", errors)
                self._check_lru_linked_list(host_lru, ct, "host", errors)

        # ── PART 4: Size Accounting ──
        for ct in self.tree_components:
            evictable = 0
            protected = 0
            for n in all_nodes:
                if n is self.root_node:
                    continue
                cd = n.component_data[ct]
                if cd.value is not None:
                    toks = len(cd.value)
                    if cd.lock_ref > 0:
                        protected += toks
                    else:
                        evictable += toks
            if self.component_evictable_size_[ct] != evictable:
                E(
                    f"[Size] {ct} evictable={self.component_evictable_size_[ct]} "
                    f"!= recomputed={evictable}"
                )
            if self.component_protected_size_[ct] != protected:
                E(
                    f"[Size] {ct} protected={self.component_protected_size_[ct]} "
                    f"!= recomputed={protected}"
                )

        # ── PART 5: Ongoing Operations ──
        for nid, n in self.ongoing_write_through.items():
            if n not in all_node_set:
                E(f"[Ongoing] write_through node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] write_through node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )
        for nid, n in self.ongoing_load_back.items():
            if n not in all_node_set:
                E(f"[Ongoing] load_back node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] load_back node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )

        # ── Result ──
        if errors:
            msg = (
                f"Sanity check FAILED ({len(errors)} violations "
                f"across {len(all_nodes)} nodes):\n"
                + "\n".join(f"  {e}" for e in errors)
            )
            logger.error(msg)
            self.pretty_print()
            raise AssertionError(msg)
        logger.debug(
            f"Sanity check PASSED: {len(all_nodes)} nodes, "
            f"{len(self.tree_components)} components"
        )

    def _check_lru_linked_list(
        self,
        lru: "UnifiedLRUList",
        ct: ComponentType,
        label: str,
        errors: list[str],
    ) -> None:
        """Walk a LRU doubly-linked list, collect integrity errors."""
        pt = lru._pt  # use LRU's own pointer slot
        visited: set[int] = set()
        x = lru.head.lru_next[pt]
        prev = lru.head
        while x is not None and x != lru.tail:
            if x.lru_prev[pt] != prev:
                errors.append(f"[{label}][{ct}] broken prev at node {x.id}")
            if x.id not in lru.cache:
                errors.append(f"[{label}][{ct}] node {x.id} in list not cache")
            if x.id in visited:
                errors.append(f"[{label}][{ct}] cycle at node {x.id}")
                break
            visited.add(x.id)
            prev = x
            x = x.lru_next[pt]
        if x is None:
            errors.append(
                f"[{label}][{ct}] broken chain: lru_next is None "
                f"after node {prev.id if hasattr(prev, 'id') else 'head'}"
            )
        if len(visited) != len(lru.cache):
            errors.append(
                f"[{label}][{ct}] list={len(visited)} != cache={len(lru.cache)}"
            )

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            component_str = " ".join(
                f"{ct}={'yes' if node.component_data[ct].value is not None else 'no'}"
                for ct in self.tree_components
            )
            print(
                " " * indent,
                f"[{node.id}]",
                len(node.key),
                f"full_lock={node.component_data[BASE_COMPONENT_TYPE].lock_ref}",
                component_str,
            )
            for child in node.children.values():
                stack.append((child, indent + 2))

    def _rebuild_host_leaf_sets(self) -> None:
        """Rebuild evictable_host_leaves after L1-only reset."""
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node is not self.root_node:
                self._update_evictable_leaf_sets(node)
            stack.extend(node.children.values())

    def _rebuild_host_lru_lists(self) -> None:
        """Rebuild host_lru_lists for extra components after L1-only reset.
        Walks the tree and adds nodes with host component data to the
        appropriate host LRU list."""
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node is not self.root_node:
                for ct in self.tree_components:
                    if ct == BASE_COMPONENT_TYPE:
                        continue  # Full uses evictable_host_leaves, not host LRU
                    cd = node.component_data[ct]
                    if cd.host_value is not None:
                        self.host_lru_lists[ct].insert_mru(node)
            stack.extend(node.children.values())
