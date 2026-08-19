# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host metadata for merged and forked pipeline groups."""

from dataclasses import dataclass, field, replace

from .enums import PipelineGroupMode
from .resources import MemoryResource, PipelineConfig


class _GroupMemberRef:
    def __init__(self, group: "PipelineGroup", member: MemoryResource) -> None:
        self.group = group
        self.member = member


@dataclass(kw_only=True, eq=False)
class PipelineGroup(MemoryResource):
    members: list[MemoryResource] = field(default_factory=list)
    mode: PipelineGroupMode = PipelineGroupMode.Merge

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("PipelineGroup requires at least one member")
        self.is_barrier = True
        for member in self.members:
            if member.pipeline_group is not None:
                raise ValueError(f"member {member.name!r} already belongs to a group")
            member.pipeline_group = self
        if self.pipeline_config is None:
            self.pipeline_config = self._derive_config()
        self._validate_config()

    def __getattr__(self, name: str) -> _GroupMemberRef:
        for member in self.__dict__.get("members", ()):
            if member.name == name:
                return _GroupMemberRef(self, member)
        raise AttributeError(name)

    @property
    def num_barriers_per_stage(self) -> int:
        return 2 if self.mode is PipelineGroupMode.FusedMerge else len(self.members) + 1

    @property
    def is_heterogeneous(self) -> bool:
        return len({m.pipeline_config.pipeline_type for m in self.members}) > 1

    def _derive_config(self) -> PipelineConfig:
        configs = [member.pipeline_config for member in self.members]
        if any(config is None for config in configs):
            raise ValueError("every PipelineGroup member needs a pipeline_config")
        first = configs[0]
        if any(config.num_stages != first.num_stages for config in configs[1:]):
            raise ValueError("PipelineGroup members have mismatched num_stages")
        if self.mode is PipelineGroupMode.Fork:
            num_bytes = sum(config.num_bytes for config in configs)
        else:
            num_bytes = max(config.num_bytes for config in configs)
        return replace(first, num_bytes=num_bytes)

    def _validate_config(self) -> None:
        config = self.pipeline_config
        for member in self.members:
            member_config = member.pipeline_config
            if member_config.num_stages != config.num_stages:
                raise ValueError("explicit group config has mismatched num_stages")
            if member_config.interleave_strides != config.interleave_strides:
                raise ValueError(
                    "PipelineGroup members have mismatched interleave strides"
                )
        if self.mode is PipelineGroupMode.FusedMerge and config.has_interleaved_stride:
            raise ValueError("FusedMerge does not support interleaved strides")
