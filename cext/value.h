// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dtype.h"
#include "enums.h"
#include "error.h"

#include <array>
#include <utility>


class UnpackedShape;

// Reference-counted heap storage for shape dimensions, shared by PackedShape and UnpackedShape.
// The dimensions are preceded by their reference count, in the same allocation.
namespace _shape_detail {
    // Returns a copy of `dims` with a reference count of 1
    int64_t* heap_dims_create(size_t ndim, const int64_t* dims);

    static inline ReferenceCounter* heap_dims_refcount(int64_t* dims) {
        return reinterpret_cast<ReferenceCounter*>(dims) - 1;
    }

    static inline void heap_dims_incref(int64_t* dims) {
        heap_dims_refcount(dims)->increment();
    }

    static inline void heap_dims_decref(int64_t* dims) {
        ReferenceCounter* refcount = heap_dims_refcount(dims);
        if (refcount->decrement()) {
            refcount->~ReferenceCounter();
            mem_free(refcount);
        }
    }
} // namespace _shape_detail

class PackedShape {
public:
    static constexpr size_t kMaxInlineDims = 6;
    static constexpr uint32_t kInlineDimBits = 20;
    static constexpr uint64_t kInlineDimMask = (1ull << kInlineDimBits) - 1;

private:
    uint64_t a_;
    uint64_t b_;

    static constexpr uint32_t kInlineNdimBits = 3;
    static constexpr uint64_t kInlineNdimMask = (1ull << kInlineNdimBits) - 1;

    int64_t* heap_data() const {
        return is_inline() ? nullptr : reinterpret_cast<int64_t*>(a_);
    }

    void decref_heap_dims() {
        if (int64_t* data = heap_data()) _shape_detail::heap_dims_decref(data);
    }

    bool is_inline() const {
        return a_ & kInlineNdimMask;
    }

    void unpack_inline_dims(int64_t out[kMaxInlineDims]) const;

    // The caller must ensure that all dimensions are positive and that the number of elements
    // doesn't overflow int64_t
    PackedShape(size_t ndim, const int64_t* dims);

public:
    PackedShape() : a_(0), b_(0) {}

    // One-dimensional shape. Its number of elements is just the length, so it can't overflow.
    explicit PackedShape(int64_t single_dim_length) : PackedShape(1, &single_dim_length) {
        CHECK(single_dim_length > 0);
    }

    PackedShape(const PackedShape& other) : a_(other.a_), b_(other.b_) {
        if (int64_t* data = heap_data()) _shape_detail::heap_dims_incref(data);
    }

    PackedShape(PackedShape&& other) : a_(other.a_), b_(other.b_) {
        other.a_ = other.b_ = 0;
    }

    PackedShape& operator= (const PackedShape& other) {
        if (this != &other) {
            PackedShape copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    PackedShape& operator= (PackedShape&& other) {
        if (this != &other) {
            decref_heap_dims();
            a_ = other.a_;
            b_ = other.b_;
            other.a_ = other.b_ = 0;
        }
        return *this;
    }

    ~PackedShape() {
        decref_heap_dims();
    }

    size_t ndim() const {
        size_t inline_ndim = a_ & kInlineNdimMask;
        return inline_ndim ? inline_ndim : b_;
    }

    // Heap-allocated dimensions are shared with the returned shape rather than copied
    UnpackedShape unpack() const;

    int64_t unpack_1d() const;

    int64_t numel() const;

    // Returns the shape that this shape and `other` broadcast to, following NumPy rules.
    // Raises an error if they are incompatible, or if the result has too many elements.
    Expected<PackedShape> broadcast(const PackedShape& other) const;

    friend class UnpackedShape;
};

static_assert(sizeof(PackedShape) == 16);


// A shape whose dimensions are directly addressable. Dimensions that PackedShape keeps on the
// heap (more than kMaxInlineDims of them, or ones too large to be packed) share its
// reference-counted heap storage, so converting between the two never copies them.
// Other dimensions are stored inline.
class UnpackedShape {
    friend class PackedShape;
    static constexpr size_t kMaxInlineDims = PackedShape::kMaxInlineDims;

    explicit UnpackedShape(const PackedShape& shape);

    // The caller must ensure that all dimensions are positive and that the number of elements
    // doesn't overflow int64_t
    UnpackedShape(size_t ndim, const int64_t* dims);

    size_t ndim_;
    int64_t* heap_data_;  // nullptr if the dimensions are stored in `inline_`
    int64_t inline_[kMaxInlineDims];
public:
    // Zero-dimensional shape
    UnpackedShape() : ndim_(0), heap_data_(nullptr) {}

    // Raises an error if the number of elements would overflow int64_t
    static Expected<UnpackedShape> create(size_t ndim, const int64_t* dims);

    // Heap-allocated dimensions are shared rather than copied
    UnpackedShape(const UnpackedShape& other) : ndim_(other.ndim_), heap_data_(other.heap_data_) {
        if (heap_data_)
            _shape_detail::heap_dims_incref(heap_data_);
        else
            mem_copy(inline_, other.inline_, ndim_ * sizeof(int64_t));
    }

    void operator=(const UnpackedShape&) = delete;

    UnpackedShape(UnpackedShape&& other) : ndim_(other.ndim_), heap_data_(other.heap_data_) {
        if (!heap_data_)
            mem_copy(inline_, other.inline_, ndim_ * sizeof(int64_t));
        other.ndim_ = 0;
        other.heap_data_ = nullptr;
    }

    UnpackedShape& operator= (UnpackedShape&& other) {
        if (this != &other) {
            if (heap_data_) _shape_detail::heap_dims_decref(heap_data_);
            ndim_ = other.ndim_;
            heap_data_ = other.heap_data_;
            if (!heap_data_)
                mem_copy(inline_, other.inline_, ndim_ * sizeof(int64_t));
            other.ndim_ = 0;
            other.heap_data_ = nullptr;
        }
        return *this;
    }

    ~UnpackedShape() {
        if (heap_data_) _shape_detail::heap_dims_decref(heap_data_);
    }

    size_t ndim() const {
        return ndim_;
    }

    const int64_t* data() const {
        return heap_data_ ? heap_data_ : inline_;
    }

    int64_t operator[] (size_t i) const {
        return data()[i];
    }

    PackedShape pack() const;

    // Returns the shape that this shape and `other` broadcast to, following NumPy rules.
    // Raises an error if they are incompatible, or if the result has too many elements.
    Expected<UnpackedShape> broadcast(const UnpackedShape& other) const;

    bool operator== (const UnpackedShape& other) const {
        if (ndim_ != other.ndim_) return false;
        const int64_t* x = data();
        const int64_t* y = other.data();
        for (size_t i = 0; i < ndim_; ++i) {
            if (x[i] != y[i]) return false;
        }
        return true;
    }

    bool operator!= (const UnpackedShape& other) const {
        return !(*this == other);
    }
};

static inline int append_to_string_builder(const UnpackedShape& us, StringBuilder* sb) {
    sb->append("(");
    for (size_t i = 0; i < us.ndim(); ++i) {
        if (i > 0) sb->append(", ");
        sb->append(us[i]);
    }
    if (us.ndim() == 1) sb->append(",");
    sb->append(")");
    return 0;
}

static inline int append_to_string_builder(const PackedShape& ps, StringBuilder* sb) {
    return append_to_string_builder(ps.unpack(), sb);
}

struct Tile;
struct Scalar;
struct Vector;
struct Pointer;
struct OpaqueValue;

struct TensorBase {
    enum Kind : uint8_t {
        kTile,  // cuda.tile.Tile
        kScalar,  // cuda.lang.Scalar
        kVector,  // cuda.lang.Vector
        kPointer,  // cuda.lang.Pointer
        kOpaque,  // cuda.lang.OpaqueValue
    };

    enum DataTag : uint8_t {
        kBits,  // scalar or "splat" vector/tile
        kBitsArray,  // pointer to an array of Bits
        kVar  // symbolic tile/scalar/vector/pointer/opaque value
    };

    using Bits = uint64_t;

    PyObject_HEAD
    union {
        Bits bits;
        // This is not an efficient way of storing data: instead of densely packing according to
        // the dtype, we always use 64 bits per element, regardless of the dtype.
        // This is fine because we're not trying to build a NumPy replacement here; we just need
        // to represent small constant vectors/tiles at compile time. This is a fairly niche case,
        // so it makes sense to optimize for simplicity rather than performance.
        Bits* bits_array;
        PyObject* var;
    } data_;  // active field is selected by `tag`
    DType dtype;
    DataTag tag_;
    Kind kind;

    TensorBase(const TensorBase&) = delete;
    void operator=(const TensorBase&) = delete;

    ~TensorBase();

    // Switches from kBits to kBitsArray with `count` zero-initialized elements
    Status allocate_bits_array(int64_t count);

    inline bool is_symbolic() const {
        return tag_ == kVar;
    }

    inline PyObject* var() const {
        CHECK(is_symbolic());
        return data_.var;
    }

    inline PackedShape shape() const;
    inline int64_t numel() const;
};

struct Tile : TensorBase {
    constexpr static Kind kind_constant = kTile;

    PackedShape shape_;

    static PyTypeObject pytype;
};

struct Scalar : TensorBase {
    constexpr static Kind kind_constant = kScalar;

    static PyTypeObject pytype;
};

struct Vector : TensorBase {
    constexpr static Kind kind_constant = kVector;

    int64_t length;

    static PyTypeObject pytype;
};

struct Pointer : TensorBase {
    constexpr static Kind kind_constant = kPointer;

    static PyTypeObject pytype;
};

// A value of an opaque dtype, e.g. mbarrier. Always symbolic.
struct OpaqueValue : TensorBase {
    constexpr static Kind kind_constant = kOpaque;

    static PyTypeObject pytype;
};

static inline void reference_add(TensorBase& tensor) {
    Py_INCREF(&tensor);
}

static inline void reference_remove(TensorBase& tensor) {
    Py_DECREF(&tensor);
}

PackedShape TensorBase::shape() const {
    switch (kind) {
    case kTile: return static_cast<const Tile*>(this)->shape_;
    case kScalar: case kPointer: case kOpaque: return PackedShape();
    case kVector:
        return PackedShape(static_cast<const Vector*>(this)->length);
    }
    CHECK_UNREACHABLE;
}

int64_t TensorBase::numel() const {
    switch (kind) {
    case kTile: return static_cast<const Tile*>(this)->shape_.numel();
    case kScalar: case kPointer: case kOpaque: return 1;
    case kVector: return static_cast<const Vector*>(this)->length;
    }
    CHECK_UNREACHABLE;
}


class Value;
struct SliceValue;
struct TupleValue;
class DeletionQueue;


// Base class for the reference counted values that a Value can point to.
// Unlike with SimpleRefcount, nested values are deleted through a DeletionQueue.
template <typename Derived>  // curiously recurring template
struct HeapValue {
    ReferenceCounter refcount;
};


// Values that are stored as a PyObject*
#define FOREACH_PYOBJECT_VALUE_TYPE(X) \
    X(kNone) /* Py_None */ \
    X(kEllipsis) /* Py_Ellipsis */ \
    X(kNotImplemented) /* Py_NotImplemented */ \
    X(kBoolConstant) /* PyBool */ \
    X(kIntConstant) /* PyLong */ \
    X(kFloatConstant) /* PyFloat */ \
    X(kTensor) \
    X(kPySlice) /* Raw PyObject* slice */ \
    X(kPyTuple) /* Raw PyObject* tuple */ \
    X(kUserDefinedEnum) \
    X(kString) /* aka PyUnicode object */ \
    X(kUnsupportedPyObject) /* Any other Python object (delayed error) */

// Parsed values that a Value can point to. They derive from HeapValue.
#define FOREACH_HEAP_VALUE_TYPE(X) \
    X(kSliceValue, SliceValue) \
    X(kTupleValue, TupleValue)

class BorrowedValue {
    friend class Value;
    friend class DeletionQueue;

    enum Tag : uint8_t {
        kUndefined = 0,

        // Store supported enums as integers
        #define ENUM_TAG_ENTRY(name, _macro, _pymod) k##name,
        FOREACH_ENUM(ENUM_TAG_ENTRY)
        #undef ENUM_TAG_ENTRY

        // Store PyObject by pointer
        #define PYOBJECT_TAG_ENUM_ENTRY(name) name,
        FOREACH_PYOBJECT_VALUE_TYPE(PYOBJECT_TAG_ENUM_ENTRY)
        #undef PYOBJECT_TAG_ENUM_ENTRY

        // Store SliceValue etc. by pointer
        #define HEAP_TAG_ENUM_ENTRY(name, _type) name,
        FOREACH_HEAP_VALUE_TYPE(HEAP_TAG_ENUM_ENTRY)
        #undef HEAP_TAG_ENUM_ENTRY
    };

public:
    // The parsed value subset of Tag, with the same values
    enum class HeapValueTag : uint8_t {
        #define HEAP_VALUE_TAG_ENUM_ENTRY(name, _type) name = Tag::name,
        FOREACH_HEAP_VALUE_TYPE(HEAP_VALUE_TAG_ENUM_ENTRY)
        #undef HEAP_VALUE_TAG_ENUM_ENTRY
    };

private:
    Tag tag_;
    void* pointer_;

    BorrowedValue(Tag tag, void* pointer) : tag_(tag), pointer_(pointer) {}

    BorrowedValue(Tag tag, uint32_t u32)
        : tag_(tag), pointer_(reinterpret_cast<void*>(static_cast<uintptr_t>(u32))) {}

    template <typename T>
    BorrowedValue(HeapValue<T>* heap_value)
        : tag_(static_cast<Tag>(T::kHeapValueTag)), pointer_(heap_value) {}

    template <typename T>
    T* pointer() const {
        return static_cast<T*>(pointer_);
    }

    PyObject* pyobject() const {
        return pointer<PyObject>();
    }

    uint32_t u32() const {
        return reinterpret_cast<uintptr_t>(pointer_);
    }

    template <typename E> struct DelayedStaticAssertFail;

    template <typename E>
    static constexpr Tag enum_tag() {
        if constexpr (false) {}
        #define FROM_ENUM_CASE(name, _macro, _pymod) \
            else if constexpr (std::is_same_v<E, name>) \
                return k##name;
        FOREACH_ENUM(FROM_ENUM_CASE)
        #undef FROM_ENUM_CASE
        else static_assert(DelayedStaticAssertFail<E>::value);
    }

public:
    static BorrowedValue from_pybool(PyObject* pyobj) {
        return BorrowedValue(kBoolConstant, pyobj);
    }

    static BorrowedValue from_pylong(PyObject* pyobj) {
        return BorrowedValue(kIntConstant, pyobj);
    }

    static BorrowedValue from_pyfloat(PyObject* pyobj) {
        return BorrowedValue(kFloatConstant, pyobj);
    }

    static BorrowedValue from_tensor(TensorBase* tensor) {
        return BorrowedValue(kTensor, tensor);
    }

    static BorrowedValue none() {
        return BorrowedValue(kNone, Py_None);
    }

    inline bool is_none() const {
        return tag_ == kNone;
    }

    static BorrowedValue ellipsis() {
        return BorrowedValue(kEllipsis, Py_Ellipsis);
    }

    inline bool is_ellipsis() const {
        return tag_ == kEllipsis;
    }

    static BorrowedValue not_implemented() {
        return BorrowedValue(kNotImplemented, Py_NotImplemented);
    }

    template <typename E>
    static BorrowedValue from_enum(E value) {
        using T = std::underlying_type_t<E>;
        static_assert(sizeof(T) <= sizeof(uint32_t));
        return BorrowedValue(enum_tag<E>(), static_cast<uint32_t>(static_cast<T>(value)));
    }

    template <typename E>
    bool is_enum() const {
        return tag_ == enum_tag<E>();
    }

    template <typename E>
    E as_enum() const {
        CHECK(is_enum<E>());
        return static_cast<E>(u32());
    }

    static BorrowedValue from_user_defined_enum(PyObject* pyobj) {
        return BorrowedValue(kUserDefinedEnum, pyobj);
    }

    static BorrowedValue from_pyslice(PyObject* pyobj) {
        return BorrowedValue(kPySlice, pyobj);
    }

    static inline BorrowedValue from_slice_value(SliceValue* slice_value);

    static BorrowedValue from_pytuple(PyObject* pyobj) {
        return BorrowedValue(kPyTuple, pyobj);
    }

    static BorrowedValue from_unsupported_pyobject(PyObject* pyobj) {
        return BorrowedValue(kUnsupportedPyObject, pyobj);
    }

    inline bool is_unsupported_pyobject() const {
        return tag_ == kUnsupportedPyObject;
    }

    static BorrowedValue from_string(PyObject* pyobj) {
        return BorrowedValue(kString, pyobj);
    }

    inline bool is_string() const {
        return tag_ == kString;
    }

    inline PyObject* as_string() const {
        CHECK(is_string());
        return pyobject();
    }

    static inline BorrowedValue from_tuple_value(TupleValue* tuple_value);

    inline NumericDTypeCategory loosely_typed_scalar_category() const {
        switch (tag_) {
        case kBoolConstant: return NumericDTypeCategory::Boolean;
        case kIntConstant: return NumericDTypeCategory::Integer;
        case kFloatConstant: return NumericDTypeCategory::Float;
        default:
            CHECK_UNREACHABLE;
        }
    }

    inline bool is_loosely_typed_scalar() const {
        return tag_ == kBoolConstant || tag_ == kFloatConstant || tag_ == kIntConstant;
    }

    inline bool is_loosely_typed_bool() const {
        return tag_ == kBoolConstant;
    }

    inline bool is_loosely_typed_int() const {
        return tag_ == kIntConstant;
    }

    inline PyObject* as_loosely_typed_scalar() const {
        CHECK(is_loosely_typed_scalar());
        return pyobject();
    }

    inline bool is_tensor() const {
        return tag_ == kTensor;
    }

    inline TensorBase* as_tensor() const {
        CHECK(is_tensor());
        return pointer<TensorBase>();
    }

    inline DType tensor_dtype() const {
        return as_tensor()->dtype;
    }

    // A Python number or a zero-dimensional tensor
    inline bool is_0d_tensor_like() const {
        return is_loosely_typed_scalar() || (is_tensor() && as_tensor()->shape().ndim() == 0);
    }

    inline bool is_integer_tensor_like() const {
        return tag_ == kIntConstant || (tag_ == kTensor && is_integer_dtype(tensor_dtype()));
    }

    inline bool is_boolean_tensor_like() const {
        return tag_ == kBoolConstant || (tag_ == kTensor && tensor_dtype() == k_bool_);
    }

    // A Python float has the default float dtype (float32), which is unrestricted
    inline bool is_unrestricted_float_tensor_like() const {
        return tag_ == kFloatConstant
            || (tag_ == kTensor && is_unrestricted_float_dtype(tensor_dtype()));
    }

    // A slice may be stored either as a Python slice object or as a SliceValue
    inline bool is_slice() const {
        return tag_ == kPySlice || tag_ == kSliceValue;
    }

    // Parses a Python slice object on demand
    RefPtr<SliceValue> as_slice() const;

    // A tuple may be stored either as a Python tuple object or as a TupleValue
    inline bool is_tuple() const {
        return tag_ == kPyTuple || tag_ == kTupleValue;
    }

    // Parses a Python tuple object on demand
    RefPtr<TupleValue> as_tuple() const;

    PyPtr to_pyobject() const;

    // Appends the value's repr(), without creating Python objects for parsed values
    int append_repr_to_builder(StringBuilder* sb) const;

    // Appends a description of the value's type, e.g. "int" or "Tile[float32, (2, 3)]".
    // To include it in a message, use use_type(), e.g. raise(..., use_type(val)).
    void append_type_name_to_builder(StringBuilder* sb, GlobalLock& lock) const;

    inline Value steal() const;

    inline Value to_owned() const;
};


class Value {
    friend class BorrowedValue;
    friend class DeletionQueue;
    #define VALUE_HEAP_VALUE_FRIEND(_name, type) friend struct type;
    FOREACH_HEAP_VALUE_TYPE(VALUE_HEAP_VALUE_FRIEND)
    #undef VALUE_HEAP_VALUE_FRIEND

    uint64_t data_;

    static uint64_t encode(BorrowedValue::Tag tag, void* pointer) {
        return (static_cast<uint64_t>(tag) << 56) | reinterpret_cast<uint64_t>(pointer);
    }

    static BorrowedValue::Tag decode_tag(uint64_t data) {
        return static_cast<BorrowedValue::Tag>(data >> 56);
    }

    template <typename T>
    static T* decode_pointer(uint64_t data) {
        return reinterpret_cast<T*>(data & 0x00ff'ffff'ffff'ffff);
    }

    Value(BorrowedValue::Tag tag, void* pointer) : data_(encode(tag, pointer)) {}

    BorrowedValue::Tag tag() const {
        return decode_tag(data_);
    }

    template <typename T>
    T* pointer() const {
        return decode_pointer<T>(data_);
    }

    PyObject* pyobject() const {
        return pointer<PyObject>();
    }

    void add_reference() const;

    void destroy();

    // Like destroy(), but SliceValues and TupleValues whose reference count drops to zero
    // are added to `queue` instead of being deleted. Leaves *this undefined.
    void release(DeletionQueue* queue);

public:
    Value() : data_(0) {}

    Value(const Value& other) : data_(other.data_) {
        add_reference();
    }

    Value(Value&& other) : data_(other.data_) {
        other.data_ = 0;
    }

    Value& operator= (const Value& other) {
        if (this != &other) {
            Value copy(other);
            *this = std::move(copy);
        }
        return *this;
    }

    Value& operator= (Value&& other) {
        if (this != &other) {
            // Release the old object only after *this is updated, since that may run arbitrary code
            Value old(std::move(*this));
            data_ = other.data_;
            other.data_ = 0;
        }
        return *this;
    }

    ~Value() {
        destroy();
    }

    BorrowedValue borrow() const {
        return BorrowedValue(tag(), pointer<void>());
    }

    PyPtr release_pyobject();

    static Value from_tensor(RefPtr<TensorBase>&& tensor) {
        return Value(BorrowedValue::kTensor, reinterpret_cast<PyObject*>(tensor.release()));
    }

    static Value not_implemented() {
        return Value(BorrowedValue::kNotImplemented, Py_NewRef(Py_NotImplemented));
    }

    bool is_not_implemented() const {
        return tag() == BorrowedValue::kNotImplemented;
    }

    static Value from_slice_value(RefPtr<SliceValue>&& slice_value) {
        return Value(BorrowedValue::kSliceValue, slice_value.release());
    }

    static Value from_tuple_value(RefPtr<TupleValue>&& tuple_value) {
        return Value(BorrowedValue::kTupleValue, tuple_value.release());
    }

    template <typename... Items>
    static inline Value pack_tuple(Items&&... items);
};

Value BorrowedValue::steal() const {
    return Value(tag_, pointer_);
}

Value BorrowedValue::to_owned() const {
    Value ret(tag_, pointer_);
    ret.add_reference();
    return ret;
}

template <typename T>
void reference_add(HeapValue<T>& obj) {
    obj.refcount.increment();
}

// Deleting is safe even for deeply nested values, because ~Value() releases nested values
// through a DeletionQueue
template <typename T>
void reference_remove(HeapValue<T>& obj) {
    if (obj.refcount.decrement()) delete static_cast<T*>(&obj);
}

// A slice whose start/stop/step are None values (see BorrowedValue::none()) if omitted
struct SliceValue : HeapValue<SliceValue> {
    static constexpr BorrowedValue::HeapValueTag kHeapValueTag =
        BorrowedValue::HeapValueTag::kSliceValue;

    Value start, stop, step;

    // Returns the corresponding Python slice object, creating it on first use
    inline PyPtr to_pyobject();

private:
    friend class BorrowedValue;  // as_slice() caches the slice object it was parsed from
    friend class DeletionQueue;
    PyPtr cached_pyobject_;

    inline PyPtr create_pyobject();

    // Releases the items into `queue`, before deleting
    void release(DeletionQueue* queue) {
        start.release(queue);
        stop.release(queue);
        step.release(queue);
    }
};

// A tuple of Values. The items are stored in the same allocation, right after the header,
// like a flexible array member in C.
struct TupleValue : HeapValue<TupleValue> {
    static constexpr BorrowedValue::HeapValueTag kHeapValueTag =
        BorrowedValue::HeapValueTag::kTupleValue;

    // Creates a tuple of `size` undefined Values
    static RefPtr<TupleValue> create(size_t size) {
        RefPtr<TupleValue> ret = allocate(size);
        Value* items = ret->items();
        for (size_t i = 0; i < size; ++i)
            new (&items[i]) Value();
        return ret;
    }

    template <typename... Items>
    static RefPtr<TupleValue> create(Items&&... items) {
        RefPtr<TupleValue> ret = allocate(sizeof...(Items));
        Value* value_items = ret->items();
        size_t i = 0;
        (new (&value_items[i++]) Value(std::forward<Items>(items)), ...);
        return ret;
    }

    TupleValue(const TupleValue&) = delete;
    void operator=(const TupleValue&) = delete;

    ~TupleValue() {
        Value* items_ptr = items();
        for (size_t i = 0; i < size_; ++i)
            items_ptr[i].~Value();
    }

    // Use create() instead. Deleting goes through operator delete below, since the memory
    // was allocated with xcalloc().
    static void* operator new(size_t) = delete;
    static void operator delete(void* p) {
        mem_free(p);
    }

    size_t size() const {
        return size_;
    }

    Value* items() {
        return reinterpret_cast<Value*>(this + 1);
    }

    const Value* items() const {
        return reinterpret_cast<const Value*>(this + 1);
    }

    Value& operator[] (size_t i) {
        return items()[i];
    }

    const Value& operator[] (size_t i) const {
        return items()[i];
    }

    // Returns the corresponding Python tuple object, creating it on first use
    inline PyPtr to_pyobject();

private:
    explicit TupleValue(size_t size) : size_(size) {}

    static RefPtr<TupleValue> allocate(size_t size) {
        void* mem = xcalloc(1, sizeof(TupleValue) + size * sizeof(Value));
        TupleValue* ret = ::new (mem) TupleValue(size);
        return steal(ret);
    }

    inline PyPtr create_pyobject();

    void release(DeletionQueue* queue) {
        for (size_t i = 0; i < size_; ++i)
            items()[i].release(queue);
    }

    friend class BorrowedValue;
    friend class DeletionQueue;
    size_t size_;
    PyPtr cached_pyobject_;
};

static_assert(sizeof(TupleValue) % alignof(Value) == 0);

BorrowedValue BorrowedValue::from_slice_value(SliceValue* slice_value) {
    return BorrowedValue(slice_value);
}

BorrowedValue BorrowedValue::from_tuple_value(TupleValue* tuple_value) {
    return BorrowedValue(tuple_value);
}

template <typename... Items>
Value Value::pack_tuple(Items&&... items) {
    return from_tuple_value(TupleValue::create(std::forward<Items>(items)...));
}

// Slices and tuples may be nested arbitrarily deep, so the conversions below are guarded
// against unbounded recursion, like CPython's repr() of containers
PyPtr SliceValue::to_pyobject() {
    if (!cached_pyobject_) {
        if (Py_EnterRecursiveCall(" while converting a slice to a Python object")) return {};
        cached_pyobject_ = create_pyobject();
        Py_LeaveRecursiveCall();
        if (!cached_pyobject_) return {};
    }
    return cached_pyobject_;
}

PyPtr SliceValue::create_pyobject() {
    PyPtr start_obj = start.borrow().to_pyobject();
    if (!start_obj) return {};
    PyPtr stop_obj = stop.borrow().to_pyobject();
    if (!stop_obj) return {};
    PyPtr step_obj = step.borrow().to_pyobject();
    if (!step_obj) return {};
    return steal(PySlice_New(start_obj.get(), stop_obj.get(), step_obj.get()));
}

PyPtr TupleValue::to_pyobject() {
    if (!cached_pyobject_) {
        if (Py_EnterRecursiveCall(" while converting a tuple to a Python object")) return {};
        cached_pyobject_ = create_pyobject();
        Py_LeaveRecursiveCall();
        if (!cached_pyobject_) return {};
    }
    return cached_pyobject_;
}

PyPtr TupleValue::create_pyobject() {
    PyPtr tuple = steal(PyTuple_New(static_cast<Py_ssize_t>(size_)));
    if (!tuple) return {};
    for (size_t i = 0; i < size_; ++i) {
        PyPtr item = items()[i].borrow().to_pyobject();
        if (!item) return {};
        PyTuple_SET_ITEM(tuple.get(), i, item.release());
    }
    return tuple;
}

Status value_module_init(PyObject* m);

// Appends the slice like Python's repr(), e.g. "slice(1, None, None)"
static inline int append_to_string_builder(const SliceValue& slice, StringBuilder* sb) {
    if (Py_EnterRecursiveCall(" while getting the repr of a slice")) return -1;
    sb->append("slice(");
    int status = slice.start.borrow().append_repr_to_builder(sb);
    if (status == 0) {
        sb->append(", ");
        status = slice.stop.borrow().append_repr_to_builder(sb);
    }
    if (status == 0) {
        sb->append(", ");
        status = slice.step.borrow().append_repr_to_builder(sb);
    }
    sb->append(")");
    Py_LeaveRecursiveCall();
    return status;
}

// Appends the tuple like Python's repr(), e.g. "(1, 2)" or "(4,)"
static inline int append_to_string_builder(const TupleValue& tuple, StringBuilder* sb) {
    if (Py_EnterRecursiveCall(" while getting the repr of a tuple")) return -1;
    sb->append("(");
    int status = 0;
    for (size_t i = 0; i < tuple.size() && status == 0; ++i) {
        if (i > 0) sb->append(", ");
        status = tuple[i].borrow().append_repr_to_builder(sb);
    }
    if (tuple.size() == 1) sb->append(",");
    sb->append(")");
    Py_LeaveRecursiveCall();
    return status;
}

// Prints the value's repr(), e.g. raise(PyExc_IndexError, "Vector index ", key, ...)
static inline int append_to_string_builder(BorrowedValue value, StringBuilder* sb) {
    return value.append_repr_to_builder(sb);
}

// Wraps a value passed to raise()/to_pyunicode()/etc. to print its type rather than the value,
// e.g. raise(PyExc_TypeError, "Shape must be a tuple, not ", use_type(val))
struct UseType {
    BorrowedValue value;
};

static inline UseType use_type(BorrowedValue value) {
    return UseType{value};
}

static inline UseType use_type(const Value& value) {
    return UseType{value.borrow()};
}

static inline int append_to_string_builder(UseType u, StringBuilder* sb) {
    GlobalLock lock;
    u.value.append_type_name_to_builder(sb, lock);
    return 0;
}
