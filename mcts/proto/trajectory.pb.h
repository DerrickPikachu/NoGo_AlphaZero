// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: trajectory.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_trajectory_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_trajectory_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3020000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3020000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_trajectory_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_trajectory_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_trajectory_2eproto;
class trajectory;
struct trajectoryDefaultTypeInternal;
extern trajectoryDefaultTypeInternal _trajectory_default_instance_;
class trajectory_transition;
struct trajectory_transitionDefaultTypeInternal;
extern trajectory_transitionDefaultTypeInternal _trajectory_transition_default_instance_;
PROTOBUF_NAMESPACE_OPEN
template<> ::trajectory* Arena::CreateMaybeMessage<::trajectory>(Arena*);
template<> ::trajectory_transition* Arena::CreateMaybeMessage<::trajectory_transition>(Arena*);
PROTOBUF_NAMESPACE_CLOSE

// ===================================================================

class trajectory_transition final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:trajectory.transition) */ {
 public:
  inline trajectory_transition() : trajectory_transition(nullptr) {}
  ~trajectory_transition() override;
  explicit PROTOBUF_CONSTEXPR trajectory_transition(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  trajectory_transition(const trajectory_transition& from);
  trajectory_transition(trajectory_transition&& from) noexcept
    : trajectory_transition() {
    *this = ::std::move(from);
  }

  inline trajectory_transition& operator=(const trajectory_transition& from) {
    CopyFrom(from);
    return *this;
  }
  inline trajectory_transition& operator=(trajectory_transition&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const trajectory_transition& default_instance() {
    return *internal_default_instance();
  }
  static inline const trajectory_transition* internal_default_instance() {
    return reinterpret_cast<const trajectory_transition*>(
               &_trajectory_transition_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(trajectory_transition& a, trajectory_transition& b) {
    a.Swap(&b);
  }
  inline void Swap(trajectory_transition* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(trajectory_transition* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  trajectory_transition* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<trajectory_transition>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const trajectory_transition& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const trajectory_transition& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to, const ::PROTOBUF_NAMESPACE_ID::Message& from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(trajectory_transition* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "trajectory.transition";
  }
  protected:
  explicit trajectory_transition(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kStateFieldNumber = 1,
    kActionIdFieldNumber = 2,
    kRewardFieldNumber = 3,
  };
  // repeated float state = 1;
  int state_size() const;
  private:
  int _internal_state_size() const;
  public:
  void clear_state();
  private:
  float _internal_state(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_state() const;
  void _internal_add_state(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_state();
  public:
  float state(int index) const;
  void set_state(int index, float value);
  void add_state(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      state() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_state();

  // int32 action_id = 2;
  void clear_action_id();
  int32_t action_id() const;
  void set_action_id(int32_t value);
  private:
  int32_t _internal_action_id() const;
  void _internal_set_action_id(int32_t value);
  public:

  // float reward = 3;
  void clear_reward();
  float reward() const;
  void set_reward(float value);
  private:
  float _internal_reward() const;
  void _internal_set_reward(float value);
  public:

  // @@protoc_insertion_point(class_scope:trajectory.transition)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > state_;
  int32_t action_id_;
  float reward_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_trajectory_2eproto;
};
// -------------------------------------------------------------------

class trajectory final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:trajectory) */ {
 public:
  inline trajectory() : trajectory(nullptr) {}
  ~trajectory() override;
  explicit PROTOBUF_CONSTEXPR trajectory(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  trajectory(const trajectory& from);
  trajectory(trajectory&& from) noexcept
    : trajectory() {
    *this = ::std::move(from);
  }

  inline trajectory& operator=(const trajectory& from) {
    CopyFrom(from);
    return *this;
  }
  inline trajectory& operator=(trajectory&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const trajectory& default_instance() {
    return *internal_default_instance();
  }
  static inline const trajectory* internal_default_instance() {
    return reinterpret_cast<const trajectory*>(
               &_trajectory_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(trajectory& a, trajectory& b) {
    a.Swap(&b);
  }
  inline void Swap(trajectory* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(trajectory* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  trajectory* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<trajectory>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const trajectory& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const trajectory& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to, const ::PROTOBUF_NAMESPACE_ID::Message& from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(trajectory* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "trajectory";
  }
  protected:
  explicit trajectory(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef trajectory_transition transition;

  // accessors -------------------------------------------------------

  enum : int {
    kTransitionsFieldNumber = 1,
  };
  // repeated .trajectory.transition transitions = 1;
  int transitions_size() const;
  private:
  int _internal_transitions_size() const;
  public:
  void clear_transitions();
  ::trajectory_transition* mutable_transitions(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::trajectory_transition >*
      mutable_transitions();
  private:
  const ::trajectory_transition& _internal_transitions(int index) const;
  ::trajectory_transition* _internal_add_transitions();
  public:
  const ::trajectory_transition& transitions(int index) const;
  ::trajectory_transition* add_transitions();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::trajectory_transition >&
      transitions() const;

  // @@protoc_insertion_point(class_scope:trajectory)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::trajectory_transition > transitions_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_trajectory_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// trajectory_transition

// repeated float state = 1;
inline int trajectory_transition::_internal_state_size() const {
  return state_.size();
}
inline int trajectory_transition::state_size() const {
  return _internal_state_size();
}
inline void trajectory_transition::clear_state() {
  state_.Clear();
}
inline float trajectory_transition::_internal_state(int index) const {
  return state_.Get(index);
}
inline float trajectory_transition::state(int index) const {
  // @@protoc_insertion_point(field_get:trajectory.transition.state)
  return _internal_state(index);
}
inline void trajectory_transition::set_state(int index, float value) {
  state_.Set(index, value);
  // @@protoc_insertion_point(field_set:trajectory.transition.state)
}
inline void trajectory_transition::_internal_add_state(float value) {
  state_.Add(value);
}
inline void trajectory_transition::add_state(float value) {
  _internal_add_state(value);
  // @@protoc_insertion_point(field_add:trajectory.transition.state)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
trajectory_transition::_internal_state() const {
  return state_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
trajectory_transition::state() const {
  // @@protoc_insertion_point(field_list:trajectory.transition.state)
  return _internal_state();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
trajectory_transition::_internal_mutable_state() {
  return &state_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
trajectory_transition::mutable_state() {
  // @@protoc_insertion_point(field_mutable_list:trajectory.transition.state)
  return _internal_mutable_state();
}

// int32 action_id = 2;
inline void trajectory_transition::clear_action_id() {
  action_id_ = 0;
}
inline int32_t trajectory_transition::_internal_action_id() const {
  return action_id_;
}
inline int32_t trajectory_transition::action_id() const {
  // @@protoc_insertion_point(field_get:trajectory.transition.action_id)
  return _internal_action_id();
}
inline void trajectory_transition::_internal_set_action_id(int32_t value) {
  
  action_id_ = value;
}
inline void trajectory_transition::set_action_id(int32_t value) {
  _internal_set_action_id(value);
  // @@protoc_insertion_point(field_set:trajectory.transition.action_id)
}

// float reward = 3;
inline void trajectory_transition::clear_reward() {
  reward_ = 0;
}
inline float trajectory_transition::_internal_reward() const {
  return reward_;
}
inline float trajectory_transition::reward() const {
  // @@protoc_insertion_point(field_get:trajectory.transition.reward)
  return _internal_reward();
}
inline void trajectory_transition::_internal_set_reward(float value) {
  
  reward_ = value;
}
inline void trajectory_transition::set_reward(float value) {
  _internal_set_reward(value);
  // @@protoc_insertion_point(field_set:trajectory.transition.reward)
}

// -------------------------------------------------------------------

// trajectory

// repeated .trajectory.transition transitions = 1;
inline int trajectory::_internal_transitions_size() const {
  return transitions_.size();
}
inline int trajectory::transitions_size() const {
  return _internal_transitions_size();
}
inline void trajectory::clear_transitions() {
  transitions_.Clear();
}
inline ::trajectory_transition* trajectory::mutable_transitions(int index) {
  // @@protoc_insertion_point(field_mutable:trajectory.transitions)
  return transitions_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::trajectory_transition >*
trajectory::mutable_transitions() {
  // @@protoc_insertion_point(field_mutable_list:trajectory.transitions)
  return &transitions_;
}
inline const ::trajectory_transition& trajectory::_internal_transitions(int index) const {
  return transitions_.Get(index);
}
inline const ::trajectory_transition& trajectory::transitions(int index) const {
  // @@protoc_insertion_point(field_get:trajectory.transitions)
  return _internal_transitions(index);
}
inline ::trajectory_transition* trajectory::_internal_add_transitions() {
  return transitions_.Add();
}
inline ::trajectory_transition* trajectory::add_transitions() {
  ::trajectory_transition* _add = _internal_add_transitions();
  // @@protoc_insertion_point(field_add:trajectory.transitions)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::trajectory_transition >&
trajectory::transitions() const {
  // @@protoc_insertion_point(field_list:trajectory.transitions)
  return transitions_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)


// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_trajectory_2eproto
