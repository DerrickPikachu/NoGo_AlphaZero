// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: trajectory.proto

#include "trajectory.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG

namespace _pb = ::PROTOBUF_NAMESPACE_ID;
namespace _pbi = _pb::internal;

PROTOBUF_CONSTEXPR trajectory_transition::trajectory_transition(
    ::_pbi::ConstantInitialized)
  : state_()
  , action_id_(0)
  , reward_(0){}
struct trajectory_transitionDefaultTypeInternal {
  PROTOBUF_CONSTEXPR trajectory_transitionDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~trajectory_transitionDefaultTypeInternal() {}
  union {
    trajectory_transition _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 trajectory_transitionDefaultTypeInternal _trajectory_transition_default_instance_;
PROTOBUF_CONSTEXPR trajectory::trajectory(
    ::_pbi::ConstantInitialized)
  : transitions_(){}
struct trajectoryDefaultTypeInternal {
  PROTOBUF_CONSTEXPR trajectoryDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~trajectoryDefaultTypeInternal() {}
  union {
    trajectory _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 trajectoryDefaultTypeInternal _trajectory_default_instance_;
static ::_pb::Metadata file_level_metadata_trajectory_2eproto[2];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_trajectory_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_trajectory_2eproto = nullptr;

const uint32_t TableStruct_trajectory_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::trajectory_transition, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::trajectory_transition, state_),
  PROTOBUF_FIELD_OFFSET(::trajectory_transition, action_id_),
  PROTOBUF_FIELD_OFFSET(::trajectory_transition, reward_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::trajectory, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::trajectory, transitions_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::trajectory_transition)},
  { 9, -1, -1, sizeof(::trajectory)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::_trajectory_transition_default_instance_._instance,
  &::_trajectory_default_instance_._instance,
};

const char descriptor_table_protodef_trajectory_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\020trajectory.proto\"y\n\ntrajectory\022+\n\013tran"
  "sitions\030\001 \003(\0132\026.trajectory.transition\032>\n"
  "\ntransition\022\r\n\005state\030\001 \003(\002\022\021\n\taction_id\030"
  "\002 \001(\005\022\016\n\006reward\030\003 \001(\002b\006proto3"
  ;
static ::_pbi::once_flag descriptor_table_trajectory_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_trajectory_2eproto = {
    false, false, 149, descriptor_table_protodef_trajectory_2eproto,
    "trajectory.proto",
    &descriptor_table_trajectory_2eproto_once, nullptr, 0, 2,
    schemas, file_default_instances, TableStruct_trajectory_2eproto::offsets,
    file_level_metadata_trajectory_2eproto, file_level_enum_descriptors_trajectory_2eproto,
    file_level_service_descriptors_trajectory_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_trajectory_2eproto_getter() {
  return &descriptor_table_trajectory_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_trajectory_2eproto(&descriptor_table_trajectory_2eproto);

// ===================================================================

class trajectory_transition::_Internal {
 public:
};

trajectory_transition::trajectory_transition(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  state_(arena) {
  SharedCtor();
  // @@protoc_insertion_point(arena_constructor:trajectory.transition)
}
trajectory_transition::trajectory_transition(const trajectory_transition& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      state_(from.state_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&action_id_, &from.action_id_,
    static_cast<size_t>(reinterpret_cast<char*>(&reward_) -
    reinterpret_cast<char*>(&action_id_)) + sizeof(reward_));
  // @@protoc_insertion_point(copy_constructor:trajectory.transition)
}

inline void trajectory_transition::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&action_id_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&reward_) -
    reinterpret_cast<char*>(&action_id_)) + sizeof(reward_));
}

trajectory_transition::~trajectory_transition() {
  // @@protoc_insertion_point(destructor:trajectory.transition)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void trajectory_transition::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void trajectory_transition::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void trajectory_transition::Clear() {
// @@protoc_insertion_point(message_clear_start:trajectory.transition)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  state_.Clear();
  ::memset(&action_id_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&reward_) -
      reinterpret_cast<char*>(&action_id_)) + sizeof(reward_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* trajectory_transition::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated float state = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedFloatParser(_internal_mutable_state(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<uint8_t>(tag) == 13) {
          _internal_add_state(::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr));
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      // int32 action_id = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 16)) {
          action_id_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // float reward = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 29)) {
          reward_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* trajectory_transition::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:trajectory.transition)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated float state = 1;
  if (this->_internal_state_size() > 0) {
    target = stream->WriteFixedPacked(1, _internal_state(), target);
  }

  // int32 action_id = 2;
  if (this->_internal_action_id() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt32ToArray(2, this->_internal_action_id(), target);
  }

  // float reward = 3;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_reward = this->_internal_reward();
  uint32_t raw_reward;
  memcpy(&raw_reward, &tmp_reward, sizeof(tmp_reward));
  if (raw_reward != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteFloatToArray(3, this->_internal_reward(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:trajectory.transition)
  return target;
}

size_t trajectory_transition::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:trajectory.transition)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated float state = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->_internal_state_size());
    size_t data_size = 4UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::_pbi::WireFormatLite::Int32Size(static_cast<int32_t>(data_size));
    }
    total_size += data_size;
  }

  // int32 action_id = 2;
  if (this->_internal_action_id() != 0) {
    total_size += ::_pbi::WireFormatLite::Int32SizePlusOne(this->_internal_action_id());
  }

  // float reward = 3;
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_reward = this->_internal_reward();
  uint32_t raw_reward;
  memcpy(&raw_reward, &tmp_reward, sizeof(tmp_reward));
  if (raw_reward != 0) {
    total_size += 1 + 4;
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData trajectory_transition::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    trajectory_transition::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*trajectory_transition::GetClassData() const { return &_class_data_; }

void trajectory_transition::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<trajectory_transition *>(to)->MergeFrom(
      static_cast<const trajectory_transition &>(from));
}


void trajectory_transition::MergeFrom(const trajectory_transition& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:trajectory.transition)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  state_.MergeFrom(from.state_);
  if (from._internal_action_id() != 0) {
    _internal_set_action_id(from._internal_action_id());
  }
  static_assert(sizeof(uint32_t) == sizeof(float), "Code assumes uint32_t and float are the same size.");
  float tmp_reward = from._internal_reward();
  uint32_t raw_reward;
  memcpy(&raw_reward, &tmp_reward, sizeof(tmp_reward));
  if (raw_reward != 0) {
    _internal_set_reward(from._internal_reward());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void trajectory_transition::CopyFrom(const trajectory_transition& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:trajectory.transition)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool trajectory_transition::IsInitialized() const {
  return true;
}

void trajectory_transition::InternalSwap(trajectory_transition* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  state_.InternalSwap(&other->state_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(trajectory_transition, reward_)
      + sizeof(trajectory_transition::reward_)
      - PROTOBUF_FIELD_OFFSET(trajectory_transition, action_id_)>(
          reinterpret_cast<char*>(&action_id_),
          reinterpret_cast<char*>(&other->action_id_));
}

::PROTOBUF_NAMESPACE_ID::Metadata trajectory_transition::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_trajectory_2eproto_getter, &descriptor_table_trajectory_2eproto_once,
      file_level_metadata_trajectory_2eproto[0]);
}

// ===================================================================

class trajectory::_Internal {
 public:
};

trajectory::trajectory(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  transitions_(arena) {
  SharedCtor();
  // @@protoc_insertion_point(arena_constructor:trajectory)
}
trajectory::trajectory(const trajectory& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      transitions_(from.transitions_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:trajectory)
}

inline void trajectory::SharedCtor() {
}

trajectory::~trajectory() {
  // @@protoc_insertion_point(destructor:trajectory)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void trajectory::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void trajectory::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void trajectory::Clear() {
// @@protoc_insertion_point(message_clear_start:trajectory)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  transitions_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* trajectory::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .trajectory.transition transitions = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_transitions(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* trajectory::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:trajectory)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .trajectory.transition transitions = 1;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_transitions_size()); i < n; i++) {
    const auto& repfield = this->_internal_transitions(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(1, repfield, repfield.GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:trajectory)
  return target;
}

size_t trajectory::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:trajectory)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .trajectory.transition transitions = 1;
  total_size += 1UL * this->_internal_transitions_size();
  for (const auto& msg : this->transitions_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData trajectory::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    trajectory::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*trajectory::GetClassData() const { return &_class_data_; }

void trajectory::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<trajectory *>(to)->MergeFrom(
      static_cast<const trajectory &>(from));
}


void trajectory::MergeFrom(const trajectory& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:trajectory)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  transitions_.MergeFrom(from.transitions_);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void trajectory::CopyFrom(const trajectory& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:trajectory)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool trajectory::IsInitialized() const {
  return true;
}

void trajectory::InternalSwap(trajectory* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  transitions_.InternalSwap(&other->transitions_);
}

::PROTOBUF_NAMESPACE_ID::Metadata trajectory::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_trajectory_2eproto_getter, &descriptor_table_trajectory_2eproto_once,
      file_level_metadata_trajectory_2eproto[1]);
}

// @@protoc_insertion_point(namespace_scope)
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::trajectory_transition*
Arena::CreateMaybeMessage< ::trajectory_transition >(Arena* arena) {
  return Arena::CreateMessageInternal< ::trajectory_transition >(arena);
}
template<> PROTOBUF_NOINLINE ::trajectory*
Arena::CreateMaybeMessage< ::trajectory >(Arena* arena) {
  return Arena::CreateMessageInternal< ::trajectory >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
