// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "value.h"
#include "cext_docstrings.h"
#include "tile_kernel.h"
#include "temporary_array.h"

#include <memory>

#if defined(_MSC_VER)
#include <intrin.h>
#endif


static PyObject* g_add_pyunicode;
static PyObject* g_sub_pyunicode;
static PyObject* g_mul_pyunicode;
static PyObject* g_wrapping_add_pyunicode;
static PyObject* g_wrapping_sub_pyunicode;
static PyObject* g_wrapping_mul_pyunicode;
static PyObject* g_truediv_pyunicode;
static PyObject* g_floordiv_pyunicode;
static PyObject* g_cdiv_pyunicode;
static PyObject* g_pow_pyunicode;
static PyObject* g_mod_pyunicode;
static PyObject* g_min_pyunicode;
static PyObject* g_max_pyunicode;
static PyObject* g_atan2_pyunicode;
static PyObject* g_eq_pyunicode;
static PyObject* g_ne_pyunicode;
static PyObject* g_lt_pyunicode;
static PyObject* g_le_pyunicode;
static PyObject* g_gt_pyunicode;
static PyObject* g_ge_pyunicode;
static PyObject* g_and__pyunicode;
static PyObject* g_or__pyunicode;
static PyObject* g_xor_pyunicode;
static PyObject* g_lshift_pyunicode;
static PyObject* g_rshift_pyunicode;
static PyObject* g_neg_pyunicode;
static PyObject* g_invert_pyunicode;


// ----- Heap-allocated shape dimensions -----

int64_t* _shape_detail::heap_dims_create(size_t ndim, const int64_t* dims) {
    CHECK(ndim <= (SIZE_MAX - sizeof(ReferenceCounter)) / sizeof(int64_t));
    void* mem = xcalloc(1, sizeof(ReferenceCounter) + ndim * sizeof(int64_t));
    ReferenceCounter* refcount = new (mem) ReferenceCounter();
    int64_t* res = reinterpret_cast<int64_t*>(refcount + 1);
    mem_copy(res, dims, ndim * sizeof(int64_t));
    return res;
}


// ----- PackedShape -----

static inline bool shape_fits_inline(size_t ndim, const int64_t* dims) {
    if (ndim > PackedShape::kMaxInlineDims)
        return false;
    for (size_t i = 0; i < ndim; ++i) {
        if (dims[i] < 0 || static_cast<uint64_t>(dims[i]) > PackedShape::kInlineDimMask)
            return false;
    }
    return true;
}

PackedShape::PackedShape(size_t ndim, const int64_t* dims) {
    if (shape_fits_inline(ndim, dims)) {
        // Accumulate the dimensions of each word, starting from the highest one
        auto dim = [dims] (size_t i) { return static_cast<uint64_t>(dims[i]); };
        uint64_t lo = 0, hi = 0;
        switch (ndim) {
            case 6: hi = (hi << kInlineDimBits) | dim(5); [[fallthrough]];
            case 5: hi = (hi << kInlineDimBits) | dim(4); [[fallthrough]];
            case 4: hi = (hi << kInlineDimBits) | dim(3); [[fallthrough]];
            case 3: lo = (lo << kInlineDimBits) | dim(2); [[fallthrough]];
            case 2: lo = (lo << kInlineDimBits) | dim(1); [[fallthrough]];
            case 1: lo = (lo << kInlineDimBits) | dim(0); break;
            case 0: break;
            default: CHECK_UNREACHABLE;
        }
        a_ = (lo << kInlineNdimBits) | ndim;
        b_ = hi;
    } else {
        a_ = reinterpret_cast<uintptr_t>(_shape_detail::heap_dims_create(ndim, dims));
        CHECK(a_ && !(a_ & kInlineNdimMask));
        b_ = ndim;
    }
}

UnpackedShape PackedShape::unpack() const {
    return UnpackedShape(*this);
}

int64_t PackedShape::unpack_1d() const {
    CHECK(ndim() == 1);
    const int64_t* data = heap_data();
    if (data) return data[0];

    uint64_t lo = a_ >> PackedShape::kInlineNdimBits;
    return static_cast<int64_t>(lo & PackedShape::kInlineDimMask);
}

void PackedShape::unpack_inline_dims(int64_t out[kMaxInlineDims]) const {
    constexpr uint64_t mask = PackedShape::kInlineDimMask;
    constexpr uint32_t bits = PackedShape::kInlineDimBits;
    uint64_t lo = a_ >> PackedShape::kInlineNdimBits;
    uint64_t hi = b_;
    out[0] = static_cast<int64_t>(lo & mask); lo >>= bits;
    out[1] = static_cast<int64_t>(lo & mask); lo >>= bits;
    out[2] = static_cast<int64_t>(lo);
    out[3] = static_cast<int64_t>(hi & mask); hi >>= bits;
    out[4] = static_cast<int64_t>(hi & mask); hi >>= bits;
    out[5] = static_cast<int64_t>(hi);
}

int64_t PackedShape::numel() const {
    int64_t buf[kMaxInlineDims];
    const int64_t* data = heap_data();
    if (!data) {
        unpack_inline_dims(buf);
        data = buf;
    }
    // All dimensions are positive, so each partial product is at most the total,
    // which can't overflow since shapes are only created with a valid number of elements
    size_t n = ndim();
    int64_t ret = 1;
    for (size_t i = 0; i < n; ++i)
        ret *= data[i];
    return ret;
}

Expected<PackedShape> PackedShape::broadcast(const PackedShape& other) const {
    if (a_ == other.a_ && b_ == other.b_)
        return *this;
    Expected<UnpackedShape> res = unpack().broadcast(other.unpack());
    if (!res.is_ok()) return std::move(res.error());
    return res->pack();
}


// ----- UnpackedShape -----

UnpackedShape::UnpackedShape(const PackedShape& shape)
        : ndim_(shape.ndim()), heap_data_(shape.heap_data()) {
    if (heap_data_) {
        _shape_detail::heap_dims_incref(heap_data_);
        return;
    }
    shape.unpack_inline_dims(inline_);
}

static PyPtr shape_to_pytuple(const UnpackedShape& shape) {
    size_t ndim = shape.ndim();
    PyPtr ret = steal(PyTuple_New(static_cast<Py_ssize_t>(ndim)));
    if (!ret) return {};
    for (size_t i = 0; i < ndim; ++i) {
        PyObject* dim = PyLong_FromLongLong(shape[i]);
        if (!dim) return {};
        PyTuple_SET_ITEM(ret.get(), i, dim);
    }
    return ret;
}

// Sets *result to a * b. Returns true if the product overflows int64_t.
static inline bool int64_mul_overflow(int64_t a, int64_t b, int64_t* result) {
#if defined(_MSC_VER)
    *result = static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
    return __mulh(a, b) != (*result >> 63);
#else
    return __builtin_mul_overflow(a, b, result);
#endif
}

// Returns the number of elements, or std::nullopt if it overflows int64_t.
// All dimensions must be positive.
static std::optional<int64_t> checked_numel(size_t ndim, const int64_t* dims) {
    int64_t ret = 1;
    for (size_t i = 0; i < ndim; ++i) {
        CHECK(dims[i] > 0);
        if (int64_mul_overflow(ret, dims[i], &ret)) return std::nullopt;
    }
    return ret;
}

Expected<UnpackedShape> UnpackedShape::create(size_t ndim, const int64_t* dims) {
    UnpackedShape ret(ndim, dims);
    for (size_t i = 0; i < ndim; ++i) {
        if (dims[i] <= 0)
            return make_error(Error::kTypeError, "Shape ", ret, " has a non-positive dimension");
    }
    if (!checked_numel(ndim, dims).has_value())
        return make_error(Error::kTypeError, "Shape ", ret, " has too many elements");
    return ret;
}

UnpackedShape::UnpackedShape(size_t ndim, const int64_t* dims) : ndim_(ndim), heap_data_(nullptr) {
    if (ndim <= kMaxInlineDims)
        mem_copy(inline_, dims, ndim * sizeof(int64_t));
    else
        heap_data_ = _shape_detail::heap_dims_create(ndim, dims);
}

PackedShape UnpackedShape::pack() const {
    if (!heap_data_)
        return PackedShape(ndim_, inline_);

    // Our heap dimensions come either from a heap PackedShape, or from a shape with more than
    // kMaxInlineDims dimensions. Either way, they can't be stored inline, so sharing them
    // yields a valid PackedShape.
    PackedShape res;
    _shape_detail::heap_dims_incref(heap_data_);
    res.a_ = reinterpret_cast<uintptr_t>(heap_data_);
    res.b_ = ndim_;
    return res;
}

Expected<UnpackedShape> UnpackedShape::broadcast(const UnpackedShape& other) const {
    // Make `x` the shape with more dimensions; `y` is aligned to its trailing dimensions
    const UnpackedShape* x = this;
    const UnpackedShape* y = &other;
    if (x->ndim_ < y->ndim_)
        std::swap(x, y);
    const int64_t* x_dims = x->data();
    const int64_t* y_dims = y->data();

    // Check compatibility, and whether the result is the same as one of the inputs
    size_t offset = x->ndim_ - y->ndim_;
    bool same_as_x = true;
    bool same_as_y = offset == 0;
    for (size_t i = 0; i < y->ndim_; ++i) {
        int64_t x_dim = x_dims[offset + i], y_dim = y_dims[i];
        if (x_dim == y_dim)
            continue;
        if (y_dim == 1)
            same_as_y = false;
        else if (x_dim == 1)
            same_as_x = false;
        else
            return make_error(Error::kTypeError, "Shapes ", *this, " and ", other,
                              " are not broadcastable");
    }

    // Reuse an existing shape if possible, to avoid allocating new heap dimensions
    if (same_as_x) return *x;
    if (same_as_y) return *y;

    // Start from a copy of `x` (inline, or in a fresh heap block that we own),
    // then replace its 1s with the corresponding dims of `y`
    UnpackedShape res(x->ndim_, x_dims);
    int64_t* dims = res.heap_data_ ? res.heap_data_ : res.inline_;
    for (size_t i = offset; i < x->ndim_; ++i) {
        if (dims[i] == 1)
            dims[i] = y_dims[i - offset];
    }
    if (!checked_numel(res.ndim_, dims).has_value())
        return make_error(Error::kTypeError, "Broadcasting ", *this, " and ", other,
                          " results in a shape with too many elements");
    return res;
}

// ----- DeletionQueue -----

// A list of heap values whose reference count has dropped to zero. Values can be nested
// arbitrarily deep, so to bound the stack depth, nested values are added to the queue
// instead of being deleted recursively, and drain() deletes them one at a time.
// The list is intrusive: a queued value's reference count, which is no longer needed,
// is replaced by the next one, encoded like Value::data_.
class DeletionQueue {
public:
    DeletionQueue() = default;

    DeletionQueue(const DeletionQueue&) = delete;
    void operator=(const DeletionQueue&) = delete;

    ~DeletionQueue() {
        CHECK(head_ == 0);
    }

    template <typename T>
    void push(HeapValue<T>* obj) {
        obj->refcount.count_ = head_;
        head_ = Value::encode(static_cast<BorrowedValue::Tag>(T::kHeapValueTag), obj);
    }

    void drain() {
        while (head_) {
            switch (static_cast<BorrowedValue::HeapValueTag>(Value::decode_tag(head_))) {
            #define DRAIN_HEAP_VALUE_CASE(name, type) \
            case BorrowedValue::HeapValueTag::name: { \
                type* obj = Value::decode_pointer<type>(head_); \
                head_ = obj->refcount.count_; \
                obj->release(this); \
                delete obj; \
                break; \
            }
            FOREACH_HEAP_VALUE_TYPE(DRAIN_HEAP_VALUE_CASE)
            #undef DRAIN_HEAP_VALUE_CASE
            }
        }
    }

private:
    uint64_t head_ = 0;
    static_assert(sizeof(size_t) == sizeof(uint64_t));
};

// ----- Value/BorrowedValue implementation -----

void Value::add_reference() const {
    using T = BorrowedValue;
    switch (tag()) {
    case T::kUndefined:
    #define ADD_REFERENCE_ENUM_CASE(name, _macro, _pymod) case T::k##name:
    FOREACH_ENUM(ADD_REFERENCE_ENUM_CASE)
    #undef ADD_REFERENCE_ENUM_CASE
        break;
    #define ADD_REFERENCE_PYOBJECT_VALUE_CASE(name) case T::name:
    FOREACH_PYOBJECT_VALUE_TYPE(ADD_REFERENCE_PYOBJECT_VALUE_CASE)
    #undef ADD_REFERENCE_PYOBJECT_VALUE_CASE
        Py_INCREF(pyobject());
        break;
    #define ADD_REFERENCE_HEAP_VALUE_CASE(name, type) \
    case T::name: \
        reference_add(*pointer<type>()); \
        break;
    FOREACH_HEAP_VALUE_TYPE(ADD_REFERENCE_HEAP_VALUE_CASE)
    #undef ADD_REFERENCE_HEAP_VALUE_CASE
    }
}

void Value::destroy() {
    DeletionQueue queue;
    release(&queue);
    queue.drain();
}

void Value::release(DeletionQueue* queue) {
    using T = BorrowedValue;
    switch (tag()) {
    case T::kUndefined:
    #define RELEASE_ENUM_CASE(name, _macro, _pymod) case T::k##name:
    FOREACH_ENUM(RELEASE_ENUM_CASE)
    #undef RELEASE_ENUM_CASE
        break;

    #define RELEASE_PYOBJECT_VALUE_CASE(name) \
    case T::name:

    FOREACH_PYOBJECT_VALUE_TYPE(RELEASE_PYOBJECT_VALUE_CASE)
    #undef RELEASE_PYOBJECT_VALUE_CASE
        Py_DECREF(pyobject());
        break;

    #define RELEASE_HEAP_VALUE_CASE(name, type) \
    case T::name: \
        if (pointer<type>()->refcount.decrement()) \
            queue->push(pointer<type>()); \
        break;

    FOREACH_HEAP_VALUE_TYPE(RELEASE_HEAP_VALUE_CASE)
    #undef RELEASE_HEAP_VALUE_CASE

    }
    data_ = 0;
}

PyPtr BorrowedValue::to_pyobject() const {
    switch (tag_) {
    case kUndefined:
        raise(PyExc_AssertionError, "Attempted to convert an undefined value to a Python object");
        return {};

    #define TO_PYOBJECT_ENUM_CASE(name, _macro, _pymod) \
        case k##name: return newref(enum_to_pyobject<name>(static_cast<name>(u32())));
    FOREACH_ENUM(TO_PYOBJECT_ENUM_CASE)
    #undef TO_PYOBJECT_ENUM_CASE

    #define TO_PYOBJECT_PYOBJECT_VALUE_CASE(name) \
    case name:

    FOREACH_PYOBJECT_VALUE_TYPE(TO_PYOBJECT_PYOBJECT_VALUE_CASE)
    #undef TO_PYOBJECT_PYOBJECT_VALUE_CASE
        return newref(pyobject());

    #define TO_PYOBJECT_HEAP_VALUE_CASE(name, type) \
    case name: \
        return pointer<type>()->to_pyobject();
    FOREACH_HEAP_VALUE_TYPE(TO_PYOBJECT_HEAP_VALUE_CASE)
    #undef TO_PYOBJECT_HEAP_VALUE_CASE
    }
    CHECK_UNREACHABLE;
}

PyPtr Value::release_pyobject() {
    switch (tag()) {
        #define TO_PYOBJECT_PYOBJECT_VALUE_CASE(name) \
        case BorrowedValue::name:
        FOREACH_PYOBJECT_VALUE_TYPE(TO_PYOBJECT_PYOBJECT_VALUE_CASE)
        #undef TO_PYOBJECT_PYOBJECT_VALUE_CASE
        {
            PyPtr ret = steal(pyobject());
            data_ = 0;
            return ret;
        }

    default: break;
    }
    return borrow().to_pyobject();
}

int BorrowedValue::append_repr_to_builder(StringBuilder* sb) const {
    switch (tag_) {
    case kUndefined:
        sb->append("<undefined>");
        return 0;

    #define APPEND_REPR_ENUM_CASE(name, _macro, _pymod) \
        case k##name: \
            sb->append_pyobject_repr(enum_to_pyobject<name>(static_cast<name>(u32()))); \
            return 0;
    FOREACH_ENUM(APPEND_REPR_ENUM_CASE)
    #undef APPEND_REPR_ENUM_CASE

    #define APPEND_REPR_PYOBJECT_VALUE_CASE(name) case name:
    FOREACH_PYOBJECT_VALUE_TYPE(APPEND_REPR_PYOBJECT_VALUE_CASE)
    #undef APPEND_REPR_PYOBJECT_VALUE_CASE
        sb->append_pyobject_repr(pyobject());
        return 0;
    #define APPEND_REPR_HEAP_VALUE_CASE(name, type) \
    case name: \
        return append_to_string_builder(*pointer<type>(), sb);
    FOREACH_HEAP_VALUE_TYPE(APPEND_REPR_HEAP_VALUE_CASE)
    #undef APPEND_REPR_HEAP_VALUE_CASE
    }
    CHECK_UNREACHABLE;
}

// Appends the tensor's type, e.g. "tile[float32, (2, 3)]" or "pointer[float32]",
// or "Tile[float32, (2, 3)]" / "Pointer[float32]" if `capitalize` is true
static void append_tensor_type(const TensorBase* tensor, bool capitalize, StringBuilder* sb) {
    switch (tensor->kind) {
    case TensorBase::kTile:
        sb->append_many(capitalize ? "Tile[" : "tile[", tensor->dtype, ", ",
                        static_cast<const Tile*>(tensor)->shape_, "]");
        break;
    case TensorBase::kScalar:
        sb->append_many(capitalize ? "Scalar[" : "scalar[", tensor->dtype, "]");
        break;
    case TensorBase::kVector:
        sb->append_many(capitalize ? "Vector[" : "vector[", tensor->dtype,
                        ", count=", static_cast<const Vector*>(tensor)->length, "]");
        break;
    case TensorBase::kPointer:
        {
            DType pointee = pointer_dtype_pointee(tensor->dtype);
            MemorySpace space = pointer_dtype_memory_space(tensor->dtype);
            bool has_params = false;
            if (pointee) {
                sb->append_many(capitalize ? "Pointer[" : "pointer[", pointee);
                has_params = true;
            } else {
                sb->append(capitalize ? "OpaquePointer" : "opaque_pointer");
            }
            if (space != MemorySpace::GENERIC) {
                sb->append_many(has_params ? ", " : "[", space);
                has_params = true;
            }
            if (has_params)
                sb->append("]");
            break;
        }
    case TensorBase::kOpaque:
        sb->append_many(capitalize ? "OpaqueValue[" : "opaque_value[", tensor->dtype, "]");
        break;
    }
}

void BorrowedValue::append_type_name_to_builder(StringBuilder* sb) const {
    switch (tag_) {
    case kUndefined:
        sb->append("<undefined>");
        break;

    #define APPEND_ENUM_NAME(name, _macro, _pymod) \
    case k##name: \
        sb->append(enum_name<name>()); \
        break;
    FOREACH_ENUM(APPEND_ENUM_NAME);
    #undef APPEND_ENUM_NAME

    case kTensor:
        append_tensor_type(as_tensor(), true, sb);
        break;
    case kNone:
    case kEllipsis:
    case kBoolConstant:
    case kIntConstant:
    case kFloatConstant:
    case kUnsupportedPyObject:
    case kNotImplemented:
    case kUserDefinedEnum:
    case kString:
        sb->append(Py_TYPE(pyobject())->tp_name);
        break;
    case kPySlice:
    case kSliceValue:
        sb->append("slice");
        break;
    case kPyTuple:
    case kTupleValue:
        sb->append("tuple");
        break;
    }
}

static BorrowedValue borrowed_value_from_pyenum(PyObject* pyobj) {
    PyTypeObject* ty = Py_TYPE(pyobj);
    #define FROM_PYENUM_CASE(name, _macro, _pymod) \
        if (ty == enum_pytype<name>()) { \
            std::optional<name> val = enum_from_pyobject<name>(pyobj); \
            if (val.has_value()) \
                return BorrowedValue::from_enum(*val); \
            else \
                return BorrowedValue::from_unsupported_pyobject(pyobj); \
        }
    FOREACH_ENUM(FROM_PYENUM_CASE)
    #undef FROM_PYENUM_CASE

    return BorrowedValue::from_user_defined_enum(pyobj);
}

static BorrowedValue borrowed_value_from_pyobject(PyObject* pyobj) {
    PyTypeObject* ty = Py_TYPE(pyobj);
    if (ty == &Tile::pytype || ty == &Scalar::pytype || ty == &Vector::pytype
            || ty == &Pointer::pytype || ty == &OpaqueValue::pytype)
        return BorrowedValue::from_tensor(reinterpret_cast<TensorBase*>(pyobj));

    if (pyobj == Py_NotImplemented)
        return BorrowedValue::not_implemented();

    if (pyobj == Py_Ellipsis)
        return BorrowedValue::ellipsis();

    if (PySlice_Check(pyobj))
        return BorrowedValue::from_pyslice(pyobj);

    if (PyTuple_CheckExact(pyobj))
        return BorrowedValue::from_pytuple(pyobj);

    std::optional<ConstantKind> ck = classify_constant(pyobj, false);
    if (ck.has_value()) {
        switch (*ck) {
        case ConstantKind::None_: return BorrowedValue::none();
        case ConstantKind::Bool: return BorrowedValue::from_pybool(pyobj);
        case ConstantKind::Int: return BorrowedValue::from_pylong(pyobj);
        case ConstantKind::Float: return BorrowedValue::from_pyfloat(pyobj);
        case ConstantKind::Enum: return borrowed_value_from_pyenum(pyobj);
        case ConstantKind::String: return BorrowedValue::from_string(pyobj);
        default: break;
        }
    }

    return BorrowedValue::from_unsupported_pyobject(pyobj);
}

RefPtr<SliceValue> BorrowedValue::as_slice() const {
    if (tag_ == kSliceValue)
        return newref(pointer<SliceValue>());
    CHECK(tag_ == kPySlice);

    PySliceObject* pyslice = reinterpret_cast<PySliceObject*>(pyobject());
    RefPtr<SliceValue> ret = ::steal(new SliceValue());
    ret->start = borrowed_value_from_pyobject(pyslice->start).to_owned();
    ret->stop = borrowed_value_from_pyobject(pyslice->stop).to_owned();
    ret->step = borrowed_value_from_pyobject(pyslice->step).to_owned();
    ret->cached_pyobject_ = newref(pyobject());
    return ret;
}

RefPtr<TupleValue> BorrowedValue::as_tuple() const {
    if (tag_ == kTupleValue)
        return newref(pointer<TupleValue>());
    CHECK(tag_ == kPyTuple);

    size_t size = static_cast<size_t>(PyTuple_GET_SIZE(pyobject()));
    RefPtr<TupleValue> ret = TupleValue::create(size);
    for (size_t i = 0; i < size; ++i)
        (*ret)[i] = borrowed_value_from_pyobject(PyTuple_GET_ITEM(pyobject(), i)).to_owned();
    ret->cached_pyobject_ = newref(pyobject());
    return ret;
}


// ----- Fixed-width arithmetic -----

using Bits = TensorBase::Bits;
using SignedBits = int64_t;
static constexpr SignedBits kSignedBitsMin = std::numeric_limits<SignedBits>::min();
static constexpr size_t MaxWidth = sizeof(Bits) * 8;

static bool signed_int_fits(SignedBits x, uint32_t width) {
    return (x >> (width - 1)) == (x >> (MaxWidth - 1));
}

static bool unsigned_int_fits(Bits x, uint32_t width) {
    return width >= MaxWidth || !(x >> width);
}

// Converts the pending Python exception into an Error of the corresponding kind, so that
// the entry point can choose the exception class. Other exceptions are left pending.
static Error error_from_python_exception() {
    Error::Tag tag;
    if (PyErr_ExceptionMatches(PyExc_ZeroDivisionError))
        tag = Error::kZeroDivisionError;
    else if (PyErr_ExceptionMatches(PyExc_OverflowError))
        tag = Error::kOverflowError;
    else if (PyErr_ExceptionMatches(PyExc_ValueError))
        tag = Error::kValueError;
    else if (PyErr_ExceptionMatches(PyExc_TypeError))
        tag = Error::kTypeError;
    else
        return ErrorRaised;

    SavedException exc = save_raised_exception();
    exc.normalize();
    PyPtr message = steal(PyObject_Str(exc.value.get()));
    if (!message) return ErrorRaised;
    return Error(tag, message.release());
}

static Error int_overflow_error(const char* op, bool is_signed, uint32_t width) {
    return make_error(Error::kOverflowError, "Overflow in ", is_signed ? "signed " : "unsigned ",
                      width, "-bit integer ", op);
}

static uint64_t bitcast_double_to_u64(double val) {
    union { double dbl; uint64_t u64; } u = {.dbl = val};
    return u.u64;
}

static Bits truncate_int_to_signed(Bits x, uint32_t to_width) {
    uint32_t shift = MaxWidth - to_width;
    return Bits(SignedBits(x << shift) >> shift);
}

static Bits truncate_int_to_unsigned(Bits x, uint32_t to_width) {
    uint32_t shift = MaxWidth - to_width;
    return (x << shift) >> shift;
}

static void append_numeric_scalar(Bits bits, DType dtype, StringBuilder* sb) {
    if (dtype == k_bool_) {
        sb->append(bits ? "True" : "False");
    } else if (is_signed_integer_dtype(dtype)) {
        sb->append(static_cast<long long>(SignedBits(bits)));
    } else if (is_unsigned_integer_dtype(dtype)) {
        sb->append(static_cast<unsigned long long>(bits));
    } else {
        CHECK(is_float_dtype(dtype));
        sb->append(decode_float(bits, dtype));
    }
}


// Converts a double to an integer dtype, truncating toward zero.
static DTypeConversionStatus double_to_int(double value, DType to, Bits* out) {
    uint64_t f64_bits = bitcast_double_to_u64(value);
    bool sign = f64_bits >> 63;
    int biased_exp = (f64_bits >> 52) & 0x7ff;
    uint64_t mantissa = f64_bits & 0x000f'ffff'ffff'ffff;

    if (biased_exp == 0x7ff)
        return mantissa ? DTypeConversionStatus::NanNotRepresentable
                        : DTypeConversionStatus::InfinityNotRepresentable;

    // The integer part of |value|. Values of 2^64 or more don't fit any integer dtype.
    int exp = biased_exp - 1023;
    uint64_t magnitude;
    if (exp < 0) {
        magnitude = 0;  // includes zero and subnormals
    } else if (exp >= 64) {
        return DTypeConversionStatus::IntegerOutOfRange;
    } else {
        uint64_t significand = mantissa | 0x0010'0000'0000'0000;
        magnitude = exp >= 52 ? significand << (exp - 52) : significand >> (52 - exp);
    }

    bool negative = sign && magnitude;
    uint32_t width = basic_dtype_bitwidth(to);
    if (is_unsigned_integer_dtype(to)) {
        if (negative || !unsigned_int_fits(magnitude, width))
            return DTypeConversionStatus::IntegerOutOfRange;
        *out = magnitude;
    } else {
        uint64_t limit = 1ull << (width - 1);
        if (negative ? magnitude > limit : magnitude >= limit)
            return DTypeConversionStatus::IntegerOutOfRange;
        *out = negative ? 0 - magnitude : magnitude;
    }
    return DTypeConversionStatus::OK;
}

static DTypeConversionStatus convert_dtype(Bits src, DType from, DType to, Bits* out) {
    if (is_integer_dtype(from) || from == k_bool_) {
        if (is_integer_dtype(to)) {
            uint32_t to_width = basic_dtype_bitwidth(to);
            *out = is_signed_integer_dtype(to)
                    ? truncate_int_to_signed(src, to_width)
                    : truncate_int_to_unsigned(src, to_width);
            return DTypeConversionStatus::OK;
        } else if (to == k_bool_) {
            *out = src != 0;
            return DTypeConversionStatus::OK;
        } else {
            CHECK(is_float_dtype(to));
            return encode_int_as_float(src, is_signed_numeric_dtype(from), to, out);
        }
    } else {
        CHECK(is_float_dtype(from));
        double f64_src = decode_float(src, from);
        if (is_integer_dtype(to)) {
            return double_to_int(f64_src, to, out);
        } else if (to == k_bool_) {
            // NaN converts to true, as in Python and NumPy
            *out = f64_src != 0.0;
            return DTypeConversionStatus::OK;
        } else {
            CHECK(is_float_dtype(to));
            return encode_float(f64_src, to, out);
        }
    }
}

static PyPtr typed_scalar_to_pyobject(Bits value, DType dtype) {
    if (is_signed_integer_dtype(dtype)) {
        return steal(PyLong_FromLongLong(SignedBits(value)));
    } else if (is_unsigned_integer_dtype(dtype)) {
        return steal(PyLong_FromUnsignedLongLong(value));
    } else if (dtype == k_bool_) {
        return newref(value ? Py_True : Py_False);
    } else if (is_float_dtype(dtype)) {
        double d = decode_float(value, dtype);
        return steal(PyFloat_FromDouble(d));
    } else {
        raise(PyExc_AssertionError, "Can't convert a value of dtype '", dtype,
              "' to a Python object");
        return {};
    }
}

static Expected<Bits> unsigned_int_add(Bits x, Bits y, uint32_t width) {
    Bits z = x + y;
    if (((x & y) | ((x | y) & ~z)) >> (width - 1))
        return int_overflow_error("addition", false, width);
    return z;
}

static Expected<Bits> signed_int_add(Bits x, Bits y, uint32_t width) {
    Bits z = x + y;
    if (((x ^ z) & (y ^ z)) >> (width - 1))
        return int_overflow_error("addition", true, width);
    return z;
}

static Expected<Bits> unsigned_int_sub(Bits x, Bits y, uint32_t width) {
    if (y > x)
        return int_overflow_error("subtraction", false, width);
    return x - y;
}

static Expected<Bits> signed_int_sub(Bits x, Bits y, uint32_t width) {
    Bits z = x - y;
    if (((x ^ y) & (x ^ z)) >> (width - 1))
        return int_overflow_error("subtraction", true, width);
    return z;
}

static Expected<Bits> signed_int_mul(Bits x, Bits y, uint32_t width,
                                     const char* op_name = "multiplication") {
    SignedBits z;
    if (int64_mul_overflow(SignedBits(x), SignedBits(y), &z) || !signed_int_fits(z, width))
        return int_overflow_error(op_name, true, width);
    return Bits(z);
}

static Expected<Bits> unsigned_int_mul(Bits x, Bits y, uint32_t width,
                                       const char* op_name = "multiplication") {
    Bits z;
#if defined(_MSC_VER)
    static_assert(sizeof(Bits) == 8);
    z = x * y;
    if (__umulh(x, y))
        return int_overflow_error(op_name, false, width);
#else
    if (__builtin_mul_overflow(x, y, &z))
        return int_overflow_error(op_name, false, width);
#endif
    if (!unsigned_int_fits(z, width))
        return int_overflow_error(op_name, false, width);
    return z;
}

static Expected<Bits> signed_int_floordiv(Bits x, Bits y, uint32_t width) {
    SignedBits a = SignedBits(x);
    SignedBits b = SignedBits(y);
    if (!b) return make_error(Error::kZeroDivisionError, "Integer division by zero");
    if (a == kSignedBitsMin && b == -1)
        return int_overflow_error("floor division", true, width);

    SignedBits q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0)))
        --q;

    if (!signed_int_fits(q, width))
        return int_overflow_error("floor division", true, width);
    return Bits(q);
}

static Expected<Bits> signed_int_cdiv(Bits x, Bits y, uint32_t width) {
    SignedBits a = SignedBits(x);
    SignedBits b = SignedBits(y);
    if (!b) return make_error(Error::kZeroDivisionError, "Integer division by zero");
    if (a == kSignedBitsMin && b == -1)
        return int_overflow_error("ceiling division", true, width);

    SignedBits q = a / b;
    if ((a % b != 0) && ((a < 0) == (b < 0)))
        ++q;

    if (!signed_int_fits(q, width))
        return int_overflow_error("ceiling division", true, width);
    return Bits(q);
}

template <typename Mul>
static Expected<Bits> int_pow_nonnegative(Bits base, Bits exp, Mul&& mul) {
    Bits result = 1;
    while (true) {
        if (exp & 1) {
            Expected<Bits> r = mul(result, base);
            if (!r.is_ok()) return std::move(r.error());
            result = *r;
        }
        exp >>= 1;
        if (!exp) return result;  // check before we multiply, so that overflow is always legit
        Expected<Bits> r = mul(base, base);
        if (!r.is_ok()) return std::move(r.error());
        base = *r;
    }
}

static Expected<Bits> signed_int_pow(Bits x, Bits y, uint32_t width) {
    SignedBits base = SignedBits(x);
    SignedBits exp = SignedBits(y);
    if (exp < 0) {
        // Python produces a float here, which is truncated toward zero when converted back
        if (base == 0)
            return make_error(Error::kZeroDivisionError,
                              "0.0 cannot be raised to a negative power");
        if (base == 1) return 1;
        if (base == -1) return Bits(exp & 1 ? -1 : 1);
        return 0;
    }
    return int_pow_nonnegative(x, y, [width] (Bits a, Bits b) {
        return signed_int_mul(a, b, width, "power");
    });
}

static Expected<Bits> unsigned_int_pow(Bits x, Bits y, uint32_t width) {
    return int_pow_nonnegative(x, y, [width] (Bits a, Bits b) {
        return unsigned_int_mul(a, b, width, "power");
    });
}

static Expected<Bits> unsigned_int_floordiv(Bits x, Bits y) {
    if (!y) return make_error(Error::kZeroDivisionError, "Integer division by zero");
    return x / y;
}

static Expected<Bits> unsigned_int_cdiv(Bits x, Bits y) {
    if (!y) return make_error(Error::kZeroDivisionError, "Integer division by zero");
    return x / y + (x % y != 0);
}

// Python-style modulo: the result has the sign of the divisor
static Expected<Bits> signed_int_mod(Bits x, Bits y) {
    SignedBits a = SignedBits(x);
    SignedBits b = SignedBits(y);
    if (!b) return make_error(Error::kZeroDivisionError, "Integer modulo by zero");
    if (b == -1) return 0;  // avoid overflow in kSignedBitsMin % -1

    SignedBits r = a % b;
    if (r && ((r < 0) != (b < 0)))
        r += b;
    return Bits(r);
}

static Expected<Bits> unsigned_int_mod(Bits x, Bits y) {
    if (!y) return make_error(Error::kZeroDivisionError, "Integer modulo by zero");
    return x % y;
}

static Expected<double> float_truediv(double x, double y) {
    if (y == 0.0) return make_error(Error::kZeroDivisionError, "Float division by zero");
    return x / y;
}

// Applies a Python binary operator (e.g. PyNumber_FloorDivide) to two floats.
// We use this for operations that would otherwise need libm functions such as fmod().
// `func` must return a new reference to a float, or nullptr with an exception set.
template <typename F>
static Expected<double> pyfloat_binary_op(double x, double y, F&& func) {
    PyPtr x_obj = steal(PyFloat_FromDouble(x));
    if (!x_obj) return ErrorRaised;
    PyPtr y_obj = steal(PyFloat_FromDouble(y));
    if (!y_obj) return ErrorRaised;
    PyPtr res = steal(func(x_obj.get(), y_obj.get()));
    if (!res) return error_from_python_exception();
    return PyFloat_AS_DOUBLE(res.get());
}

static Expected<double> float_floordiv(double x, double y) {
    if (y == 0.0) return make_error(Error::kZeroDivisionError, "Float floor division by zero");
    return pyfloat_binary_op(x, y, PyNumber_FloorDivide);
}

static Expected<double> float_cdiv(double x, double y) {
    Expected<double> q = float_floordiv(-x, y);
    if (!q.is_ok()) return std::move(q.error());
    return -*q;
}

static Expected<double> float_mod(double x, double y) {
    if (y == 0.0) return make_error(Error::kZeroDivisionError, "Float modulo by zero");
    return pyfloat_binary_op(x, y, PyNumber_Remainder);
}

static Expected<double> float_pow(double x, double y) {
    return pyfloat_binary_op(x, y, [] (PyObject* a, PyObject* b) -> PyObject* {
        PyPtr res = steal(PyNumber_Power(a, b, Py_None));
        // Python returns a complex number for a negative base and a fractional exponent
        if (res && !PyFloat_Check(res.get())) {
            raise(PyExc_ValueError, "Negative number cannot be raised to a fractional power");
            return nullptr;
        }
        return res.release();
    });
}

static PyPtr pynumber_cdiv(PyObject* x, PyObject* y) {
    PyPtr neg_x = steal(PyNumber_Negative(x));
    if (!neg_x) return {};
    PyPtr q = steal(PyNumber_FloorDivide(neg_x.get(), y));
    if (!q) return {};
    return steal(PyNumber_Negative(q.get()));
}

static PyPtr pynumber_atan2(PyObject* x, PyObject* y) {
    static LazyImporter atan2_func;
    PyObject* func = atan2_func.get("math", "atan2");
    if (!func) return {};
    return steal(PyObject_CallFunctionObjArgs(func, x, y, nullptr));
}

static Expected<double> float_atan2(double x, double y) {
    return pyfloat_binary_op(x, y, [&] (PyObject* a, PyObject* b) {
        return pynumber_atan2(a, b).release();
    });
}

// FIXME: For now, mimics the semantics of Python's builtin min()/max().
// Need to actually respect propagate_nan.
static PyPtr pynumber_min_max(PyObject* x, PyObject* y, bool is_max) {
    int y_wins = PyObject_RichCompareBool(y, x, is_max ? Py_GT : Py_LT);
    if (y_wins < 0) return {};
    return newref(y_wins ? y : x);
}


TensorBase::~TensorBase() {
    switch (tag_) {
    case kBits: break;
    case kBitsArray: mem_free(data_.bits_array); break;
    case kVar: Py_XDECREF(data_.var); break;
    }
}


// ----- Tensors: construction and destruction ----

Status TensorBase::allocate_bits_array(int64_t count) {
    CHECK(tag_ == kBits);
    if (count <= 0)
        return raise(PyExc_AssertionError, "Tensor element count must be positive");
    static_assert(SIZE_MAX >= INT64_MAX);  // so that casting a positive count is safe
    void* mem = PyMem_RawCalloc(static_cast<size_t>(count), sizeof(Bits));
    if (!mem) {
        PyErr_NoMemory();
        return ErrorRaised;
    }
    data_.bits_array = static_cast<Bits*>(mem);
    tag_ = kBitsArray;
    return OK;
}

template <typename T>
void tensor_dealloc(PyObject* self) {
    T* t = reinterpret_cast<T*>(self);
    t->~T();
    Py_TYPE(self)->tp_free(self);
}

template <typename T>
RefPtr<T> tensor_new() {
    static_assert(std::is_base_of_v<TensorBase, T>);

    PyObject* obj = T::pytype.tp_alloc(&T::pytype, 0);
    if (!obj) return {};

    T* ret = reinterpret_cast<T*>(obj);
    ret->tag_ = TensorBase::kBits;
    ret->kind = T::kind_constant;
    return steal(ret);
}

static RefPtr<TensorBase> tensor_new_scalar_or_0d_tile(DType dtype, Bits value, bool use_tile) {
    RefPtr<TensorBase> ret;
    if (use_tile)
        ret = tensor_new<Tile>();
    else
        ret = tensor_new<Scalar>();
    if (!ret) return {};
    ret->data_.bits = value;
    ret->dtype = dtype;
    return ret;
}

static RefPtr<Vector> vector_zeros(DType dtype, int64_t length) {
    RefPtr<Vector> ret = tensor_new<Vector>();
    if (!ret) return {};
    ret->dtype = dtype;
    ret->length = length;
    ret->data_.bits = 0;
    return ret;
}

// Returns a Scalar, a Pointer or an OpaqueValue, depending on the dtype. The caller must set
// the data.
static RefPtr<TensorBase> new_0d_tensor(DType dtype) {
    RefPtr<TensorBase> ret;
    if (is_numeric_dtype(dtype)) {
        ret = tensor_new<Scalar>();
    } else if (is_pointer_dtype(dtype)) {
        ret = tensor_new<Pointer>();
    } else if (is_opaque_dtype(dtype)) {
        ret = tensor_new<OpaqueValue>();
    } else {
        raise(PyExc_AssertionError, "Unexpected dtype of a 0-d tensor: ", dtype);
        return {};
    }
    if (!ret) return {};
    ret->dtype = dtype;
    return ret;
}

static RefPtr<TensorBase> tensor_zeros_like(TensorBase* src, DType dtype) {
    RefPtr<TensorBase> ret;
    switch (src->kind) {
    case TensorBase::kTile:
        {
            RefPtr<Tile> tile = tensor_new<Tile>();
            if (!tile) return {};
            tile->shape_ = static_cast<Tile*>(src)->shape_;
            ret = std::move(tile);
            break;
        }
    case TensorBase::kVector:
        return vector_zeros(dtype, static_cast<Vector*>(src)->length);
    case TensorBase::kScalar:
    case TensorBase::kPointer:
    case TensorBase::kOpaque:
        // The kind follows the dtype, e.g. comparing two pointers yields a bool_ scalar
        if (!(ret = new_0d_tensor(dtype)))
            return {};
        break;
    }
    ret->dtype = dtype;
    ret->data_.bits = 0;
    return ret;
}

static RefPtr<TensorBase> tensor_symbolic_like(TensorBase* src, DType dtype, PyPtr&& var) {
    RefPtr<TensorBase> ret = tensor_zeros_like(src, dtype);
    if (!ret) return {};
    ret->data_.var = var.release();
    ret->tag_ = TensorBase::kVar;
    return ret;
}

static RefPtr<TensorBase> tensor_zeros_with_shape(DType dtype, const UnpackedShape& shape,
                                                  bool use_tile) {
    RefPtr<TensorBase> ret;
    if (use_tile) {
        RefPtr<Tile> tile = tensor_new<Tile>();
        if (!tile) return {};
        tile->shape_ = shape.pack();
        ret = std::move(tile);
    } else if (shape.ndim() == 0) {
        if (!(ret = new_0d_tensor(dtype)))
            return {};
    } else if (shape.ndim() == 1) {
        return vector_zeros(dtype, shape[0]);
    } else {
        raise(PyExc_AssertionError, "cuda.lang does not support tensors of shape ", shape);
        return {};
    }
    ret->dtype = dtype;
    ret->data_.bits = 0;
    return ret;
}

// `var` must have a tensor-like type, e.g. TileTy
static Expected<RefPtr<TensorBase>> make_symbolic_tensor(PyObject* var, bool use_tile) {
    PyPtr ty = steal(PyObject_CallMethod(var, "get_type", nullptr));
    if (!ty) return ErrorRaised;

    PyPtr dtype_obj = steal(PyObject_CallMethod(ty.get(), "tensor_dtype", nullptr));
    if (!dtype_obj) return ErrorRaised;
    std::optional<DType> dtype = dtype_from_pyobject(dtype_obj.get());
    if (!dtype.has_value())
        return raise(PyExc_AssertionError, "tensor_dtype() must return a DType, not ",
                     Py_TYPE(dtype_obj.get())->tp_name);

    PyPtr shape_obj = steal(PyObject_CallMethod(ty.get(), "tensor_shape", nullptr));
    if (!shape_obj) return ErrorRaised;
    if (!PyTuple_Check(shape_obj.get()))
        return raise(PyExc_AssertionError, "tensor_shape() must return a tuple, not ",
                     Py_TYPE(shape_obj.get())->tp_name);
    size_t ndim = static_cast<size_t>(PyTuple_GET_SIZE(shape_obj.get()));
    TemporaryArray<int64_t, PackedShape::kMaxInlineDims> dims_buf;
    int64_t* dims = dims_buf.allocate(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        dims[i] = PyLong_AsLongLong(PyTuple_GET_ITEM(shape_obj.get(), i));
        if (dims[i] == -1 && PyErr_Occurred()) return ErrorRaised;
    }
    Expected<UnpackedShape> shape = UnpackedShape::create(ndim, dims);
    if (!shape.is_ok()) return std::move(shape.error());

    RefPtr<TensorBase> res = tensor_zeros_with_shape(*dtype, *shape, use_tile);
    if (!res) return ErrorRaised;
    res->data_.var = Py_NewRef(var);
    res->tag_ = TensorBase::kVar;
    return res;
}

static PyObject* py_make_symbolic_tensor(PyObject*, PyObject* args) {
    PyObject* var;
    int use_tile;
    if (!PyArg_ParseTuple(args, "Op:_make_symbolic_tensor", &var, &use_tile))
        return nullptr;

    Result<RefPtr<TensorBase>> res = to_result(make_symbolic_tensor(var, use_tile));
    if (!res.is_ok()) return nullptr;
    return reinterpret_cast<PyObject*>(res->release());
}



// ----- Tensors: printing & materialization to ir.Var -----

template <typename Leaf, typename Descend, typename Ascend>
static Status walk_tensor(const UnpackedShape& shape, Leaf&& leaf, Descend&& descend,
                          Ascend&& ascend) {
    size_t ndim = shape.ndim();
    if (ndim == 0)
        return leaf(0, 0);

    TemporaryArray<size_t, PackedShape::kMaxInlineDims> index_buf;
    size_t* index = index_buf.allocate(ndim);
    if (!descend(0, 0)) return ErrorRaised;
    index[0] = 0;
    size_t d = 0;
    size_t elem_index = 0;
    while (true) {
        if (index[d] < static_cast<size_t>(shape[d])) {
            if (d + 1 == ndim) {
                if (!leaf(elem_index++, index[d])) return ErrorRaised;
                ++index[d];
            } else {
                if (!descend(d + 1, index[d])) return ErrorRaised;
                ++d;
                index[d] = 0;
            }
        } else {
            if (!ascend()) return ErrorRaised;
            if (d == 0) break;
            --d;
            ++index[d];
        }
    }
    return OK;
}

static PyPtr splat_to_nested_tuple(const UnpackedShape& shape, PyObject* elem) {
    PyPtr ret = newref(elem);
    for (size_t d = shape.ndim(); d-- > 0;) {
        size_t n = static_cast<size_t>(shape[d]);
        PyPtr level = steal(PyTuple_New(static_cast<Py_ssize_t>(n)));
        if (!level) return {};
        for (size_t i = 0; i < n; ++i)
            PyTuple_SET_ITEM(level.get(), i, Py_NewRef(ret.get()));
        ret = std::move(level);
    }
    return ret;
}

static PyPtr bits_array_to_nested_tuple(const UnpackedShape& shape, DType dtype,
                                        const Bits* data) {
    size_t ndim = shape.ndim();
    if (ndim == 0)
        return typed_scalar_to_pyobject(data[0], dtype);

    // Each tuple is owned by its parent as soon as it is created, so `tuples` holds borrowed
    // references. On failure, releasing `root` frees everything, since unfilled items are NULL.
    TemporaryArray<PyObject*, PackedShape::kMaxInlineDims> tuples_buf;
    PyObject** tuples = tuples_buf.allocate(ndim);
    PyPtr root;

    auto leaf = [&] (size_t elem_index, size_t index_in_parent) -> Status {
        PyPtr elem = typed_scalar_to_pyobject(data[elem_index], dtype);
        if (!elem) return ErrorRaised;
        PyTuple_SET_ITEM(tuples[ndim - 1], index_in_parent, elem.release());
        return OK;
    };
    auto descend = [&] (size_t depth, size_t index_in_parent) -> Status {
        PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(shape[depth]));
        if (!tuple) return ErrorRaised;
        if (depth == 0)
            root = steal(tuple);
        else
            PyTuple_SET_ITEM(tuples[depth - 1], index_in_parent, tuple);
        tuples[depth] = tuple;
        return OK;
    };
    auto ascend = [] () -> Status { return OK; };

    if (!walk_tensor(shape, leaf, descend, ascend)) return {};
    return root;
}

static PyPtr tensor_value_to_pyobject(TensorBase* tensor) {
    UnpackedShape shape = tensor->shape().unpack();
    switch (tensor->tag_) {
    case TensorBase::kBits: {
        PyPtr elem = typed_scalar_to_pyobject(tensor->data_.bits, tensor->dtype);
        if (!elem) return {};
        return splat_to_nested_tuple(shape, elem.get());
    }
    case TensorBase::kBitsArray: {
        return bits_array_to_nested_tuple(shape, tensor->dtype, tensor->data_.bits_array);
    }
    case TensorBase::kVar:
        raise(PyExc_AssertionError,
              "Value must be constant, but is only known at run time");
        return {};
    }
    CHECK_UNREACHABLE;
}

// Prints the value of a constant numeric tensor as a number or as nested lists,
// e.g. "[[1, 2], [3, 4]]"
static void append_tensor_value(const TensorBase* tensor, StringBuilder* sb) {
    CHECK(!tensor->is_symbolic());
    auto leaf = [&] (size_t elem_index, size_t index_in_parent) -> Status {
        if (index_in_parent > 0) sb->append(", ");
        Bits bits = tensor->tag_ == TensorBase::kBits
                ? tensor->data_.bits : tensor->data_.bits_array[elem_index];
        append_numeric_scalar(bits, tensor->dtype, sb);
        return OK;
    };
    auto descend = [&] (size_t, size_t index_in_parent) -> Status {
        if (index_in_parent > 0) sb->append(", ");
        sb->append("[");
        return OK;
    };
    auto ascend = [&] () -> Status {
        sb->append("]");
        return OK;
    };

    // The callbacks never fail; errors are recorded in the StringBuilder
    Status s = walk_tensor(tensor->shape().unpack(), leaf, descend, ascend);
    CHECK(s);
}


// Symbolic tensors are printed like the corresponding IR types, e.g. "<tile[float32, (2, 3)]>".
// Constant tensors also include their value, e.g. "<scalar[int32] = 5>".
static PyObject* tensor_repr(PyObject* self) {
    TensorBase* tensor = reinterpret_cast<TensorBase*>(self);
    StringBuilder sb;
    sb.append("<");
    append_tensor_type(tensor, false, &sb);

    if (!tensor->is_symbolic()) {
        if (is_numeric_dtype(tensor->dtype)) {
            sb.append(" = ");
            append_tensor_value(tensor, &sb);
        } else {
            sb.append(" (constant)");
        }
    }
    sb.append(">");
    return sb.build().release();
}


static PyPtr tensor_to_ir_var(TensorBase* tensor) {
    if (tensor->tag_ == TensorBase::kVar)
        return newref(tensor->data_.var);

    static LazyImporter const_func;
    PyObject* func = const_func.get(
            "cuda.tile._ir.core_ops", "strictly_typed_const_tensorlike");
    if (!func) return {};

    PyObject* dtype = dtype_to_pyobject(tensor->dtype);
    if (!dtype) return {};

    // A splat is passed as a single value, which is all that the lowering of constant vectors
    // in cuda.lang supports
    PyPtr value = tensor->tag_ == TensorBase::kBits
            ? typed_scalar_to_pyobject(tensor->data_.bits, tensor->dtype)
            : tensor_value_to_pyobject(tensor);
    if (!value) return {};

    PyPtr shape = shape_to_pytuple(tensor->shape().unpack());
    if (!shape) return {};

    return steal(PyObject_CallFunctionObjArgs(func, value.get(), dtype, shape.get(), nullptr));
}

static PyObject* py_tensor_to_ir_var(PyObject*, PyObject* arg) {
    BorrowedValue val = borrowed_value_from_pyobject(arg);
    if (!val.is_tensor()) {
        raise(PyExc_TypeError, "Argument must be a Tile, Scalar, Vector or Pointer, not ",
              use_type(val));
        return nullptr;
    }
    return tensor_to_ir_var(val.as_tensor()).release();
}


// ----- Tensors: type conversion -----

struct TypedOrLooseScalarValue {
    PyObject* pynumber;
    Bits bits;
    DType dtype;

    TypedOrLooseScalarValue(PyObject* pynumber) : pynumber(pynumber) {}
    TypedOrLooseScalarValue(Bits bits, DType dtype) : pynumber(nullptr), bits(bits), dtype(dtype) {}
};

static int append_to_string_builder(const TypedOrLooseScalarValue& val, StringBuilder* sb) {
    if (val.pynumber)
        sb->append(val.pynumber);
    else
        append_numeric_scalar(val.bits, val.dtype, sb);
    return 0;
}

static inline Error dtype_conversion_status_to_error(
        DTypeConversionStatus s,
        const TypedOrLooseScalarValue& src_value,
        DType target_dtype,
        const char* context = "") {
    using S = DTypeConversionStatus;
    switch (s) {
    case S::OK: CHECK_UNREACHABLE;
    case S::NegativeNotRepresentable:
        return make_error(Error::kOverflowError, context, "Cannot represent negative value ",
                          src_value, " as ", target_dtype, ", which is unsigned");
    case S::NanNotRepresentable:
        if (is_float_dtype(target_dtype)) {
            return make_error(Error::kValueError, context, "Cannot represent NaN as ",
                              target_dtype, ", which has no NaN encoding");
        } else {
            return make_error(Error::kValueError, context, "Cannot convert NaN to non-float dtype ",
                              target_dtype);
        }
    case S::InfinityNotRepresentable:
        return make_error(Error::kOverflowError, context, "Cannot convert infinity to ",
                          target_dtype);
    case S::IntegerOutOfRange:
        return make_error(Error::kOverflowError, context,
                "Value ", src_value, " is out of range for ", target_dtype);
    }
    CHECK_UNREACHABLE;
}

// `func` may return either Expected<Bits> (if it can fail) or plain Bits
template <typename F>
static Expected<RefPtr<TensorBase>> tensor_elementwise_apply(TensorBase* src, DType res_dtype,
                                                             F&& func) {
    RefPtr<TensorBase> ret = tensor_zeros_like(src, res_dtype);
    if (!ret) return ErrorRaised;

    if (src->tag_ == TensorBase::kBits) {
        auto r = func(src->data_.bits);
        if constexpr (std::is_same_v<decltype(r), Expected<Bits>>) {
            if (!r.is_ok()) return std::move(r.error());
            ret->data_.bits = *r;
        } else {
            ret->data_.bits = r;
        }
    } else {
        CHECK(src->tag_ == TensorBase::kBitsArray);
        size_t numel = static_cast<size_t>(src->numel());
        if (!ret->allocate_bits_array(src->numel())) return ErrorRaised;
        Bits* data = ret->data_.bits_array;
        const Bits* src_data = src->data_.bits_array;
        for (size_t i = 0; i < numel; ++i) {
            auto r = func(src_data[i]);
            if constexpr (std::is_same_v<decltype(r), Expected<Bits>>) {
                if (!r.is_ok()) return std::move(r.error());
                data[i] = *r;
            } else {
                data[i] = r;
            }
        }
    }
    return ret;
}

// Constants are only folded when rounding to nearest even. Otherwise, the conversion is emitted
// even for a constant tensor.
static Expected<RefPtr<TensorBase>> tensor_astype(
        TensorBase* tensor, DType to, std::optional<RoundingMode> rounding_mode = {}) {
    DType from = tensor->dtype;
    if (from == to)
        return newref(tensor);

    if (!is_numeric_dtype(from))
        return make_error(Error::kTypeError, "Cannot convert non-numeric dtype ", from, " to ", to);
    if (!is_numeric_dtype(to))
        return make_error(Error::kTypeError, "Cannot convert ", from, " to non-numeric dtype ", to);

    bool can_fold = !rounding_mode.has_value() || *rounding_mode == RoundingMode::RN;
    if (tensor->is_symbolic() || !can_fold) {
        PyPtr var = tensor_to_ir_var(tensor);
        if (!var) return ErrorRaised;

        static LazyImporter astype_func;
        PyObject* func = astype_func.get(
                "cuda.tile._ir.arithmetic_ops", "add_astype_operation");
        if (!func) return ErrorRaised;

        PyObject* to_dtype = dtype_to_pyobject(to);
        if (!to_dtype) return ErrorRaised;
        PyObject* py_rounding_mode = rounding_mode.has_value()
                ? enum_to_pyobject(*rounding_mode) : Py_None;
        if (!py_rounding_mode) return ErrorRaised;

        PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
                func, var.get(), to_dtype, py_rounding_mode, nullptr));
        if (!result_var) return ErrorRaised;

        RefPtr<TensorBase> res = tensor_symbolic_like(tensor, to, std::move(result_var));
        if (!res) return ErrorRaised;
        return res;
    }
    return tensor_elementwise_apply(tensor, to,
            [from, to] (Bits x) -> Expected<Bits> {
                Bits res;
                DTypeConversionStatus s = convert_dtype(x, from, to, &res);
                if (s == DTypeConversionStatus::OK) return res;
                return dtype_conversion_status_to_error(s, {x, from}, to);
            });
}

static Result<Bits> truncate_pylong(PyObject* pylong, DType to) {
    static_assert(sizeof(unsigned long long) == sizeof(Bits));
    Bits bits = PyLong_AsUnsignedLongLongMask(pylong);
    if (bits == Bits(-1) && PyErr_Occurred()) return ErrorRaised;
    uint32_t width = basic_dtype_bitwidth(to);
    return is_signed_integer_dtype(to) ? truncate_int_to_signed(bits, width)
                                       : truncate_int_to_unsigned(bits, width);
}

static Result<DTypeConversionStatus> convert_loosely_typed_scalar(BorrowedValue x, DType to,
                                                                  bool truncate_integers,
                                                                  Bits* out) {
    CHECK(is_numeric_dtype(to));
    PyObject* pyval = x.as_loosely_typed_scalar();
    using C = NumericDTypeCategory;
    C src_category = x.loosely_typed_scalar_category();
    switch (numeric_dtype_category(to)) {
    case C::Boolean:
        {
            // NaN converts to true, as in Python and NumPy
            int r = PyObject_IsTrue(pyval);
            if (r < 0) return ErrorRaised;
            *out = r;
            return DTypeConversionStatus::OK;
        }
    case C::Integer:
        switch (src_category) {
        case C::Boolean:
        case C::Integer:
            if (truncate_integers) {
                Result<Bits> r = truncate_pylong(pyval, to);
                if (!r.is_ok()) return ErrorRaised;
                *out = *r;
                return DTypeConversionStatus::OK;
            } else if (is_signed_integer_dtype(to)) {
                int overflow;
                SignedBits val = pylong_as_overflow_and<SignedBits>(pyval, &overflow);
                if (PyErr_Occurred()) return ErrorRaised;
                if (overflow || !signed_int_fits(val, basic_dtype_bitwidth(to)))
                    return DTypeConversionStatus::IntegerOutOfRange;
                *out = Bits(val);
            } else {
                int overflow;
                Bits val = pylong_as_overflow_and<Bits>(pyval, &overflow);
                if (PyErr_Occurred()) return ErrorRaised;
                if (overflow || !unsigned_int_fits(val, basic_dtype_bitwidth(to)))
                    return DTypeConversionStatus::IntegerOutOfRange;
                *out = val;
            }
            return DTypeConversionStatus::OK;
        case C::Float:
            return double_to_int(PyFloat_AS_DOUBLE(pyval), to, out);
        }
    case C::Float:
        switch (src_category) {
        case C::Boolean:
        case C::Integer:
            return encode_pylong_as_float(pyval, to, out);
        case C::Float:
            return encode_float(PyFloat_AS_DOUBLE(pyval), to, out);
        }
    }
    CHECK_UNREACHABLE;
}

static Result<DTypeConversionStatus> convert_loosely_typed_scalar_to_tensor(
        BorrowedValue x, DType to, bool truncate_integers, bool use_tile, RefPtr<TensorBase>* out) {
    Bits bits;
    Result<DTypeConversionStatus> s = convert_loosely_typed_scalar(x, to, truncate_integers, &bits);
    if (s.is_ok() && *s == DTypeConversionStatus::OK) {
        *out = tensor_new_scalar_or_0d_tile(to, bits, use_tile);
        if (!*out) return ErrorRaised;
    }
    return s;
}

static Expected<DType> loosely_typed_scalar_default_dtype(BorrowedValue loose_scalar) {
    PyObject* pyval = loose_scalar.as_loosely_typed_scalar();
    switch (loose_scalar.loosely_typed_scalar_category()) {
    case NumericDTypeCategory::Boolean:
        return k_bool_;
    case NumericDTypeCategory::Integer:
        {
            int overflow;
            int64_t i64_val = pylong_as_overflow_and<int64_t>(pyval, &overflow);
            if (PyErr_Occurred()) return ErrorRaised;
            if (overflow == 0)
                return (i64_val >= INT32_MIN && i64_val <= INT32_MAX) ? k_int32 : k_int64;
            if (overflow > 0) {
                pylong_as_overflow_and<uint64_t>(pyval, &overflow);
                if (PyErr_Occurred()) return ErrorRaised;
                if (overflow == 0) return k_uint64;
            }
            return make_error(Error::kOverflowError, "Integer constant ", pyval,
                              " is out of range for any supported integer dtype");
        }
    case NumericDTypeCategory::Float:
        return k_float32;
    }
    CHECK_UNREACHABLE;
}

static Expected<RefPtr<TensorBase>> loosely_typed_scalar_to_default_dtype(
            BorrowedValue x, bool use_tile) {
    Expected<DType> dtype = loosely_typed_scalar_default_dtype(x);
    if (!dtype.is_ok()) return std::move(dtype.error());
    RefPtr<TensorBase> tensor;
    Result<DTypeConversionStatus> r = convert_loosely_typed_scalar_to_tensor(
            x, *dtype, false, use_tile, &tensor);
    if (!r.is_ok()) return ErrorRaised;
    // Conversion to default dtype should never result in an error.
    // We could CHECK() here but it would be nice to avoid a hard crash in case we have a logic
    // error, since this assumption is relatively fragile.
    if (*r != DTypeConversionStatus::OK)
        return raise(PyExc_AssertionError, "Logic error: failed to convert loosely typed scalar"
                     " to its default dtype. Please report a bug.");
    return tensor;
}

// Explicit conversion between numeric dtypes, e.g. `int8(x)` or `x.astype(int8)`.
// `use_tile` determines the kind of the result if `x` is loosely typed.
static Expected<Value> tensorlike_astype(BorrowedValue x, DType to,
                                         std::optional<RoundingMode> rounding_mode,
                                         bool use_tile) {
    if (!is_numeric_dtype(to))
        return make_error(Error::kTypeError, "Cannot convert to non-numeric dtype ", to);

    RefPtr<TensorBase> tensor;
    if (x.is_loosely_typed_scalar()) {
        if (!rounding_mode.has_value() || *rounding_mode == RoundingMode::RN) {
            Result<DTypeConversionStatus> s = convert_loosely_typed_scalar_to_tensor(
                    x, to, true, use_tile, &tensor);
            if (!s.is_ok()) return ErrorRaised;
            if (*s != DTypeConversionStatus::OK)
                return dtype_conversion_status_to_error(*s, x.as_loosely_typed_scalar(), to);
            return Value::from_tensor(std::move(tensor));
        }
        // The conversion is emitted from the default dtype of the loosely typed scalar
        Expected<RefPtr<TensorBase>> r = loosely_typed_scalar_to_default_dtype(x, use_tile);
        if (!r.is_ok()) return std::move(r.error());
        tensor = std::move(*r);
    } else if (x.is_tensor()) {
        tensor = newref(x.as_tensor());
    } else {
        return make_error(Error::kTypeError, "Cannot convert ", use_type(x), " to ", to);
    }

    Expected<RefPtr<TensorBase>> res = tensor_astype(tensor.get(), to, rounding_mode);
    if (!res.is_ok()) return std::move(res.error());
    return Value::from_tensor(std::move(*res));
}

static PyObject* py_make_constant_0d_tensor(PyObject*, PyObject* args) {
    PyObject *value, *dtype_obj;
    int use_tile;
    if (!PyArg_ParseTuple(args, "OOp:_make_constant_0d_tensor", &value, &dtype_obj, &use_tile))
        return nullptr;

    std::optional<DType> dtype = dtype_from_pyobject(dtype_obj);
    if (!dtype.has_value()) {
        raise(PyExc_TypeError, "dtype must be a DType, not ", Py_TYPE(dtype_obj)->tp_name);
        return nullptr;
    }

    BorrowedValue val = borrowed_value_from_pyobject(value);
    if (!val.is_loosely_typed_scalar()) {
        raise(PyExc_TypeError, "Value must be a Python number, not ", use_type(val));
        return nullptr;
    }
    Result<Value> res = to_result(tensorlike_astype(val, *dtype, std::nullopt, use_tile));
    if (!res.is_ok()) return nullptr;
    return res->borrow().to_pyobject().release();
}

static Status optional_rounding_mode_from_pyobject(PyObject* obj,
                                                   std::optional<RoundingMode>* rounding_mode) {
    if (obj == Py_None) {
        *rounding_mode = std::nullopt;
        return OK;
    }
    *rounding_mode = enum_from_pyobject<RoundingMode>(obj);
    if (!rounding_mode->has_value())
        return raise(PyExc_ValueError, "Unknown rounding mode ", use_repr(obj));
    return OK;
}

static PyObject* release_pyobject_or_raise(Expected<Value>&& res, bool compile_time_errors) {
    if (!res.is_ok()) {
        res.error().raise(compile_time_errors);
        return nullptr;
    }
    return res->release_pyobject().release();
}

template <typename T,
          typename = typename std::enable_if<std::is_base_of_v<TensorBase, T>>::type>
static PyObject* release_pyobject_or_raise(Expected<RefPtr<T>>&& res, bool compile_time_errors) {
    if (!res.is_ok()) {
        res.error().raise(compile_time_errors);
        return nullptr;
    }
    return reinterpret_cast<PyObject*>(res->release());
}

static PyObject* py_tensorlike_astype(PyObject*, PyObject* args) {
    PyObject *x, *dtype_obj, *py_rounding_mode;
    int use_tile;
    if (!PyArg_ParseTuple(args, "OOOp:_tensorlike_astype",
                          &x, &dtype_obj, &py_rounding_mode, &use_tile))
        return nullptr;

    std::optional<DType> dtype = dtype_from_pyobject(dtype_obj);
    if (!dtype.has_value()) {
        raise(PyExc_TypeError, "dtype must be a DType, not ", Py_TYPE(dtype_obj)->tp_name);
        return nullptr;
    }
    std::optional<RoundingMode> rounding_mode;
    if (!optional_rounding_mode_from_pyobject(py_rounding_mode, &rounding_mode))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    return release_pyobject_or_raise(
            tensorlike_astype(x_val, *dtype, rounding_mode, use_tile), true);
}

static Result<DType> parse_float_dtype(PyObject* dtype_obj) {
    if (!PyObject_TypeCheck(dtype_obj, &DTypeObject::pytype))
        return raise(PyExc_TypeError, "Expected a DType object, got ", use_repr(dtype_obj));
    DType dtype = py_unwrap<DTypeObject>(dtype_obj).dtype;
    if (!is_float_dtype(dtype))
        return raise(PyExc_TypeError, "Expected a floating-point dtype, got ", dtype);
    return dtype;
}

static PyObject* py_encode_float(PyObject* module_self,
                                 PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        raise(PyExc_TypeError, "Expected 2 positional arguments");
        return nullptr;
    }
    PyObject *py_value = args[0], *py_dtype = args[1];
    double value = PyFloat_AsDouble(py_value);
    if (PyErr_Occurred()) return nullptr;
    Result<DType> dtype = parse_float_dtype(py_dtype);
    if (!dtype.is_ok()) return nullptr;
    Bits bits;
    DTypeConversionStatus s = encode_float(value, *dtype, &bits);
    if (s != DTypeConversionStatus::OK) {
        dtype_conversion_status_to_error(s, py_value, *dtype).raise_builtin_exception();
        return nullptr;
    }
    return PyLong_FromUnsignedLongLong(bits);
}

static PyObject* py_decode_float(PyObject* module_self,
                                 PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        raise(PyExc_TypeError, "Expected 2 positional arguments");
        return nullptr;
    }
    PyObject *py_bits = args[0], *py_dtype = args[1];
    if (!PyLong_Check(py_bits)) {
        raise(PyExc_TypeError, "Expected an int, got ", use_repr(py_bits));
        return nullptr;
    }
    Result<DType> dtype = parse_float_dtype(py_dtype);
    if (!dtype.is_ok()) return nullptr;
    int overflow;
    uint64_t bits = pylong_as_overflow_and<uint64_t>(py_bits, &overflow);
    if (PyErr_Occurred()) return nullptr;
    uint32_t bitwidth = float_dtype_format(*dtype).bitwidth;
    if (overflow || (bitwidth < 64 && (bits >> bitwidth))) {
        raise(PyExc_ValueError, "Bits ", use_repr(py_bits), " do not fit in ", bitwidth,
              " bits of ", *dtype);
        return nullptr;
    }
    return PyFloat_FromDouble(decode_float(bits, *dtype));
}

// Like Python's int() and float(), also accepts a string, e.g. float32("-inf")
static Expected<Value> dtype_constructor(BorrowedValue x, DType to, bool use_tile) {
    if (x.is_string()) {
        PyPtr number;
        switch (numeric_dtype_category(to)) {
        case NumericDTypeCategory::Boolean:
            return make_error(Error::kTypeError, "Cannot convert a string to ", to);
        case NumericDTypeCategory::Integer:
            number = steal(PyLong_FromUnicodeObject(x.as_string(), 10));
            break;
        case NumericDTypeCategory::Float:
            number = steal(PyFloat_FromString(x.as_string()));
            break;
        }
        if (!number) {
            if (!PyErr_ExceptionMatches(PyExc_ValueError)) return ErrorRaised;
            PyErr_Clear();
            return make_error(Error::kTypeError, "Cannot convert string ",
                              use_repr(x.as_string()), " to ", to);
        }
        BorrowedValue number_val = borrowed_value_from_pyobject(number.get());
        return tensorlike_astype(number_val, to, std::nullopt, use_tile);
    }

    if (!x.is_0d_tensor_like())
        return make_error(Error::kTypeError, "Argument of ", to,
                          "() must be a scalar or a string, not ", use_type(x));
    return tensorlike_astype(x, to, std::nullopt, use_tile);
}

static PyObject* py_dtype_constructor(PyObject*, PyObject* args) {
    PyObject *x, *dtype_obj;
    int use_tile;
    if (!PyArg_ParseTuple(args, "OOp:_dtype_constructor", &x, &dtype_obj, &use_tile))
        return nullptr;

    std::optional<DType> dtype = dtype_from_pyobject(dtype_obj);
    if (!dtype.has_value()) {
        raise(PyExc_TypeError, "dtype must be a DType, not ", Py_TYPE(dtype_obj)->tp_name);
        return nullptr;
    }

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    return release_pyobject_or_raise(dtype_constructor(x_val, *dtype, use_tile), true);
}


// ----- Tensors: type promotion & broadcasting helpers -----

// Returns the loose scalar converted to the common dtype, updates `tensor` in-place
static Expected<RefPtr<TensorBase>> promote_tensor_and_loosely_typed_scalar(
        RefPtr<TensorBase>* tensor, BorrowedValue loose_scalar,
        bool force_float) {
    DType tensor_dtype = (*tensor)->dtype;
    CHECK(is_numeric_dtype(tensor_dtype));

    DType common_dtype = tensor_dtype;
    switch (numeric_dtype_category(tensor_dtype)) {
    case NumericDTypeCategory::Boolean:
    case NumericDTypeCategory::Integer:
        if (force_float) {
            common_dtype = k_float32;
        } else if (tensor_dtype == k_bool_ || loose_scalar.loosely_typed_scalar_category()
                                                  == NumericDTypeCategory::Float) {
            Expected<DType> dtype = loosely_typed_scalar_default_dtype(loose_scalar);
            if (!dtype.is_ok()) return std::move(dtype.error());
            common_dtype = *dtype;
        }
        break;
    case NumericDTypeCategory::Float:
        break;
    }

    RefPtr<TensorBase> scalar;
    Result<DTypeConversionStatus> r = convert_loosely_typed_scalar_to_tensor(
            loose_scalar, common_dtype, false, (*tensor)->kind == TensorBase::kTile, &scalar);
    if (!r.is_ok()) return ErrorRaised;
    if (*r != DTypeConversionStatus::OK)
        return dtype_conversion_status_to_error(
                *r, loose_scalar.as_loosely_typed_scalar(), common_dtype);

    if (common_dtype != tensor_dtype) {
        Expected<RefPtr<TensorBase>> converted = tensor_astype(tensor->get(), common_dtype);
        if (!converted.is_ok()) return std::move(converted.error());
        *tensor = std::move(*converted);
    }
    return scalar;
}


static Error promote_tensors(
        RefPtr<TensorBase>* x, RefPtr<TensorBase>* y, bool force_float) {
    DType x_dtype = (*x)->dtype, y_dtype = (*y)->dtype;
    std::optional<DType> common_dtype_res = promote_dtypes(x_dtype, y_dtype);
    if (!common_dtype_res.has_value())
        return make_error(Error::kTypeError, "Implicit promotion of ", x_dtype, " and ", y_dtype,
                          " is not supported. Please perform an explicit cast instead.");

    DType common_dtype = *common_dtype_res;
    if (force_float && !is_float_dtype(common_dtype))
        common_dtype = k_float32;

    if (x_dtype != common_dtype) {
        Expected<RefPtr<TensorBase>> converted = tensor_astype(x->get(), common_dtype);
        if (!converted.is_ok()) return std::move(converted.error());
        *x = std::move(*converted);
    }

    if (y_dtype != common_dtype) {
        Expected<RefPtr<TensorBase>> converted = tensor_astype(y->get(), common_dtype);
        if (!converted.is_ok()) return std::move(converted.error());
        *y = std::move(*converted);
    }

    return OK;
}

// A pointer can only be cast implicitly to the generic memory space, and to an opaque pointer
// or a pointer with the same pointee dtype. An opaque pointer can't become a concrete one.
static bool is_implicit_pointer_cast_ok(DType from, DType to) {
    MemorySpace from_space = pointer_dtype_memory_space(from);
    MemorySpace to_space = pointer_dtype_memory_space(to);
    if (from_space != to_space && to_space != MemorySpace::GENERIC)
        return false;

    DType from_pointee = pointer_dtype_pointee(from);
    DType to_pointee = pointer_dtype_pointee(to);
    if (!from_pointee)
        return !to_pointee;
    return !to_pointee || from_pointee == to_pointee;
}

// Casts a Python number or a tensor to `dtype`, following the rules of implicit_cast() in
// cast_ops.py. A Python number becomes a scalar, or a 0-d tile if `use_tile` is true.
// Constant tensors are converted directly, symbolic ones emit a cast.
static Expected<RefPtr<TensorBase>> implicit_cast(BorrowedValue val, DType dtype, bool use_tile) {
    if (val.is_loosely_typed_scalar()) {
        if (!is_numeric_dtype(dtype))
            return make_error(Error::kValueError, "Cannot implicitly cast ",
                              val.as_loosely_typed_scalar(), " to non-numeric dtype ", dtype);
        NumericDTypeCategory from_cat = val.loosely_typed_scalar_category();
        NumericDTypeCategory to_cat = numeric_dtype_category(dtype);
        if (from_cat > to_cat) {
            // Integers 0 and 1 can be implicitly cast to bool_
            if (!val.is_loosely_typed_int() || dtype != k_bool_)
                return make_error(Error::kTypeError, "Cannot implicitly cast ",
                                  use_type(val), " to ", dtype);
            int overflow;
            int64_t i64_val = pylong_as_overflow_and<int64_t>(
                    val.as_loosely_typed_scalar(), &overflow);
            if (PyErr_Occurred()) return ErrorRaised;
            if (overflow || (i64_val != 0 && i64_val != 1))
                return make_error(Error::kTypeError, "Cannot implicitly cast ", val, " to bool_");
        }

        RefPtr<TensorBase> tensor;
        Result<DTypeConversionStatus> s = convert_loosely_typed_scalar_to_tensor(
                val, dtype, false, use_tile, &tensor);
        if (!s.is_ok()) return ErrorRaised;
        if (*s == DTypeConversionStatus::OK) return tensor;
        return dtype_conversion_status_to_error(
                *s, val.as_loosely_typed_scalar(), dtype, "Invalid implicit cast: ");
    }

    CHECK(val.is_tensor());
    TensorBase* tensor = val.as_tensor();
    DType from = tensor->dtype;
    if (from == dtype)
        return newref(tensor);

    if (is_numeric_dtype(from) && is_numeric_dtype(dtype)) {
        std::optional<DType> common_dtype = promote_dtypes(from, dtype);
        if (common_dtype.has_value() && *common_dtype == dtype)
            return tensor_astype(tensor, dtype);
    } else if (is_pointer_dtype(from) && is_pointer_dtype(dtype)
               && is_implicit_pointer_cast_ok(from, dtype)) {
        PyPtr var = tensor_to_ir_var(tensor);
        if (!var) return ErrorRaised;
        PyObject* dtype_obj = dtype_to_pyobject(dtype);
        if (!dtype_obj) return ErrorRaised;

        static LazyImporter cast_func;
        PyObject* func = cast_func.get("cuda.tile._ir.cast_ops", "implicit_cast");
        if (!func) return ErrorRaised;
        PyPtr result_var = steal(PyObject_CallFunction(
                func, "OOs", var.get(), dtype_obj, "Implicit pointer cast"));
        if (!result_var) return ErrorRaised;

        RefPtr<TensorBase> res = tensor_symbolic_like(tensor, dtype, std::move(result_var));
        if (!res) return ErrorRaised;
        return res;
    }

    return make_error(Error::kTypeError, "Cannot implicitly cast ", from, " to ", dtype);
}

static Error promote_operands(RefPtr<TensorBase>* x_tensor, RefPtr<TensorBase>* y_tensor,
                              BorrowedValue x, BorrowedValue y, bool force_float) {
    if (!*x_tensor) {
        CHECK(*y_tensor);
        Expected<RefPtr<TensorBase>> promoted =
                promote_tensor_and_loosely_typed_scalar(y_tensor, x, force_float);
        if (!promoted.is_ok()) return std::move(promoted.error());
        *x_tensor = std::move(*promoted);
    } else if (!*y_tensor) {
        Expected<RefPtr<TensorBase>> promoted =
                promote_tensor_and_loosely_typed_scalar(x_tensor, y, force_float);
        if (!promoted.is_ok()) return std::move(promoted.error());
        *y_tensor = std::move(*promoted);
    } else {
        return promote_tensors(x_tensor, y_tensor, force_float);
    }
    return OK;
}

static Error broadcast_to_unchecked(RefPtr<TensorBase>* tensor, const UnpackedShape& shape) {
    TensorBase* src = tensor->get();

    if (src->kind == TensorBase::kPointer)
        return make_error(Error::kTypeError, "Pointer cannot be broadcast to shape ", shape);
    if (src->kind == TensorBase::kOpaque)
        return make_error(Error::kTypeError, "OpaqueValue cannot be broadcast to shape ", shape);

    RefPtr<TensorBase> ret = tensor_zeros_with_shape(
            src->dtype, shape, src->kind == TensorBase::kTile);
    if (!ret) return ErrorRaised;

    switch (src->tag_) {
    case TensorBase::kBits:
        ret->data_.bits = src->data_.bits;
        break;
    case TensorBase::kBitsArray:
        {
            UnpackedShape src_shape = src->shape().unpack();
            size_t ndim = shape.ndim();
            size_t extra = ndim - src_shape.ndim();
            TemporaryArray<size_t, 2 * PackedShape::kMaxInlineDims> buf;
            size_t* src_strides = buf.allocate(2 * ndim);
            size_t* index = src_strides + ndim;
            size_t stride = 1;
            for (size_t i = ndim; i-- > 0;) {
                index[i] = 0;
                size_t src_dim = i >= extra ? static_cast<size_t>(src_shape[i - extra]) : 1;
                src_strides[i] = src_dim == 1 ? 0 : stride;
                stride *= src_dim;
            }

            size_t numel = static_cast<size_t>(ret->numel());
            const Bits* src_data = src->data_.bits_array;
            if (!ret->allocate_bits_array(ret->numel())) return ErrorRaised;
            Bits* data = ret->data_.bits_array;
            size_t src_offset = 0;
            for (size_t n = 0; n < numel; ++n) {
                data[n] = src_data[src_offset];
                for (size_t i = ndim; i-- > 0;) {
                    src_offset += src_strides[i];
                    if (++index[i] < static_cast<size_t>(shape[i])) break;
                    src_offset -= index[i] * src_strides[i];
                    index[i] = 0;
                }
            }
            break;
        }
    case TensorBase::kVar:
        {
            static LazyImporter broadcast_func;
            PyObject* func = broadcast_func.get(
                    "cuda.tile._ir.arithmetic_ops", "broadcast_to");
            if (!func) return ErrorRaised;

            PyPtr py_shape = shape_to_pytuple(shape);
            if (!py_shape) return ErrorRaised;

            PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
                    func, src->var(), py_shape.get(), nullptr));
            if (!result_var) return ErrorRaised;

            ret->data_.var = result_var.release();
            ret->tag_ = TensorBase::kVar;
            break;
        }
    }

    *tensor = std::move(ret);
    return OK;
}

static Error broadcast_operands(RefPtr<TensorBase>* x_tensor, RefPtr<TensorBase>* y_tensor) {
    UnpackedShape x_shape = (*x_tensor)->shape().unpack();
    UnpackedShape y_shape = (*y_tensor)->shape().unpack();
    Expected<UnpackedShape> common_shape = x_shape.broadcast(y_shape);
    if (!common_shape.is_ok()) return std::move(common_shape.error());

    if (x_shape != *common_shape) {
        Error err = broadcast_to_unchecked(x_tensor, *common_shape);
        if (!err.is_ok()) return err;
    }
    if (y_shape != *common_shape) {
        Error err = broadcast_to_unchecked(y_tensor, *common_shape);
        if (!err.is_ok()) return err;
    }
    return OK;
}


// ----- Tensors: binary elementwise -----

// `func` may return either Expected<Bits> (if it can fail) or plain Bits
template <typename F, typename X, typename Y>
static Error elementwise_binary_loop(Bits* out, size_t numel, F&& func, X&& x, Y&& y) {
    for (size_t i = 0; i < numel; ++i) {
        auto r = func(x(i), y(i));
        if constexpr (std::is_same_v<decltype(r), Expected<Bits>>) {
            if (!r.is_ok()) return std::move(r.error());
            out[i] = *r;
        } else {
            out[i] = r;
        }
    }
    return OK;
}

// `func` may return either Expected<Bits> (if it can fail) or plain Bits
template <typename F>
static Expected<RefPtr<TensorBase>> tensor_elementwise_apply_binary(
        TensorBase* x, TensorBase* y, DType res_dtype, F&& func) {
    RefPtr<TensorBase> ret = tensor_zeros_like(x, res_dtype);
    if (!ret) return ErrorRaised;

    if (x->tag_ == TensorBase::kBits && y->tag_ == TensorBase::kBits) {
        auto r = func(x->data_.bits, y->data_.bits);
        if constexpr (std::is_same_v<decltype(r), Expected<Bits>>) {
            if (!r.is_ok()) return std::move(r.error());
            ret->data_.bits = *r;
        } else {
            ret->data_.bits = r;
        }
        return ret;
    }

    size_t numel = static_cast<size_t>(x->numel());
    if (!ret->allocate_bits_array(x->numel())) return ErrorRaised;
    Bits* data = ret->data_.bits_array;

    Error err = OK;
    if (x->tag_ == TensorBase::kBits && y->tag_ == TensorBase::kBitsArray) {
        Bits x_val = x->data_.bits;
        const Bits* y_data = y->data_.bits_array;
        err = elementwise_binary_loop(data, numel, func,
                [&](size_t){return x_val;}, [&](size_t i){return y_data[i];});
    } else if (x->tag_ == TensorBase::kBitsArray && y->tag_ == TensorBase::kBits) {
        const Bits* x_data = x->data_.bits_array;
        Bits y_val = y->data_.bits;
        err = elementwise_binary_loop(data, numel, func,
                [&](size_t i){return x_data[i];}, [&](size_t){return y_val;});
    } else {
        CHECK(x->tag_ == TensorBase::kBitsArray);
        CHECK(y->tag_ == TensorBase::kBitsArray);
        const Bits* x_data = x->data_.bits_array;
        const Bits* y_data = y->data_.bits_array;
        err = elementwise_binary_loop(data, numel, func,
                [&](size_t i){return x_data[i];}, [&](size_t i){return y_data[i];});
    }
    if (!err.is_ok()) return std::move(err);
    return ret;
}

#define FOREACH_BINARY_ARITHMETIC_OP(X) \
    X(Add, add, "+") \
    X(Sub, sub, "-") \
    X(Mul, mul, "*") \
    X(TrueDiv, truediv, "/") \
    X(FloorDiv, floordiv, "//") \
    X(CDiv, cdiv, "cdiv") \
    X(Mod, mod, "%") \
    X(Pow, pow, "**") \
    X(Min, min, "min") \
    X(Max, max, "max") \
    X(Atan2, atan2, "atan2")

enum class BinaryArithmeticOp {
    #define BINARY_OP_ENUM_ENTRY(name, _fn, _op) name,
    FOREACH_BINARY_ARITHMETIC_OP(BINARY_OP_ENUM_ENTRY)
    #undef BINARY_OP_ENUM_ENTRY
};

static const char* binary_arithmetic_op_str(BinaryArithmeticOp binop) {
    switch (binop) {
    #define BINARY_OP_STR_CASE(name, _fn, op) \
        case BinaryArithmeticOp::name: return op;
    FOREACH_BINARY_ARITHMETIC_OP(BINARY_OP_STR_CASE);
    #undef BINARY_OP_STR_CASE
    }
    CHECK_UNREACHABLE;
}

// Integer arithmetic modulo 2**bitwidth
#define FOREACH_BINARY_WRAPPING_OP(X) \
    X(Add, wrapping_add) \
    X(Sub, wrapping_sub) \
    X(Mul, wrapping_mul)

enum class BinaryWrappingOp {
    #define WRAPPING_OP_ENUM_ENTRY(name, _fn) name,
    FOREACH_BINARY_WRAPPING_OP(WRAPPING_OP_ENUM_ENTRY)
    #undef WRAPPING_OP_ENUM_ENTRY
};

static const char* binary_wrapping_op_str(BinaryWrappingOp op) {
    switch (op) {
    #define WRAPPING_OP_STR_CASE(name, fn) \
        case BinaryWrappingOp::name: return #fn;
    FOREACH_BINARY_WRAPPING_OP(WRAPPING_OP_STR_CASE);
    #undef WRAPPING_OP_STR_CASE
    }
    CHECK_UNREACHABLE;
}

// The values match Python's rich comparison opcodes (Py_LT etc.)
#define FOREACH_COMPARISON_OP(X) \
    X(Lt, lt, Py_LT, "<") \
    X(Le, le, Py_LE, "<=") \
    X(Eq, eq, Py_EQ, "==") \
    X(Ne, ne, Py_NE, "!=") \
    X(Gt, gt, Py_GT, ">") \
    X(Ge, ge, Py_GE, ">=")

enum class ComparisonOp {
    #define COMPARISON_OP_ENUM_ENTRY(name, _fn, py_op, _op) name = py_op,
    FOREACH_COMPARISON_OP(COMPARISON_OP_ENUM_ENTRY)
    #undef COMPARISON_OP_ENUM_ENTRY
};

static const char* comparison_op_str(ComparisonOp op) {
    switch (op) {
    #define COMPARISON_OP_STR_CASE(name, _fn, _py_op, op_str) \
        case ComparisonOp::name: return op_str;
    FOREACH_COMPARISON_OP(COMPARISON_OP_STR_CASE);
    #undef COMPARISON_OP_STR_CASE
    }
    CHECK_UNREACHABLE;
}

#define FOREACH_BINARY_BITWISE_OP(X) \
    X(And, and_, "&") \
    X(Or, or_, "|") \
    X(Xor, xor, "^")

enum class BinaryBitwiseOp {
    #define BITWISE_OP_ENUM_ENTRY(name, _fn, _op) name,
    FOREACH_BINARY_BITWISE_OP(BITWISE_OP_ENUM_ENTRY)
    #undef BITWISE_OP_ENUM_ENTRY
};

static const char* binary_bitwise_op_str(BinaryBitwiseOp op) {
    switch (op) {
    #define BITWISE_OP_STR_CASE(name, _fn, op_str) \
        case BinaryBitwiseOp::name: return op_str;
    FOREACH_BINARY_BITWISE_OP(BITWISE_OP_STR_CASE);
    #undef BITWISE_OP_STR_CASE
    }
    CHECK_UNREACHABLE;
}

#define FOREACH_BITWISE_SHIFT_OP(X) \
    X(LShift, lshift, "<<") \
    X(RShift, rshift, ">>")

enum class BitwiseShiftOp {
    #define SHIFT_OP_ENUM_ENTRY(name, _fn, _op) name,
    FOREACH_BITWISE_SHIFT_OP(SHIFT_OP_ENUM_ENTRY)
    #undef SHIFT_OP_ENUM_ENTRY
};

static const char* bitwise_shift_op_str(BitwiseShiftOp op) {
    switch (op) {
    #define SHIFT_OP_STR_CASE(name, _fn, op_str) \
        case BitwiseShiftOp::name: return op_str;
    FOREACH_BITWISE_SHIFT_OP(SHIFT_OP_STR_CASE);
    #undef SHIFT_OP_STR_CASE
    }
    CHECK_UNREACHABLE;
}


static Result<Value> pointer_add_sub_offset(TensorBase* ptr, TensorBase* offset,
                                            BinaryArithmeticOp op) {
    static LazyImporter add_func;
    static LazyImporter sub_func;
    PyObject* func;
    switch (op) {
    case BinaryArithmeticOp::Add:
        func = add_func.get("cuda.lang._ir.ops", "pointer_add");
        break;
    case BinaryArithmeticOp::Sub:
        func = sub_func.get("cuda.lang._ir.ops", "pointer_sub");
        break;
    default:
        CHECK_UNREACHABLE;
    }
    if (!func) return ErrorRaised;

    PyPtr ptr_var = tensor_to_ir_var(ptr);
    if (!ptr_var) return ErrorRaised;

    PyPtr offset_var = tensor_to_ir_var(offset);
    if (!offset_var) return ErrorRaised;

    PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
            func, ptr_var.get(), offset_var.get(), nullptr));
    if (!result_var) return ErrorRaised;

    RefPtr<TensorBase> ret = tensor_symbolic_like(ptr, ptr->dtype, std::move(result_var));
    if (!ret) return ErrorRaised;
    return Value::from_tensor(std::move(ret));
}


// Computes `x op y` elementwise. Both tensors must be constant, with the same dtype and shape.
static Expected<RefPtr<TensorBase>> constant_tensor_arith(TensorBase* x, TensorBase* y,
                                                          BinaryArithmeticOp op) {
    using B = BinaryArithmeticOp;
    DType dtype = x->dtype;
    CHECK(y->dtype == dtype);

    auto apply = [&] (auto&& func) {
        return tensor_elementwise_apply_binary(x, y, dtype, func);
    };

    if (is_signed_integer_dtype(dtype)) {
        uint32_t width = basic_dtype_bitwidth(dtype);
        switch (op) {
        case B::Add: return apply([=] (Bits a, Bits b) { return signed_int_add(a, b, width); });
        case B::Sub: return apply([=] (Bits a, Bits b) { return signed_int_sub(a, b, width); });
        case B::Mul: return apply([=] (Bits a, Bits b) { return signed_int_mul(a, b, width); });
        case B::FloorDiv:
            return apply([=] (Bits a, Bits b) { return signed_int_floordiv(a, b, width); });
        case B::CDiv:
            return apply([=] (Bits a, Bits b) { return signed_int_cdiv(a, b, width); });
        case B::TrueDiv: break;
        case B::Mod: return apply(signed_int_mod);
        case B::Pow: return apply([=] (Bits a, Bits b) { return signed_int_pow(a, b, width); });
        case B::Min:
            return apply([] (Bits a, Bits b) { return SignedBits(b) < SignedBits(a) ? b : a; });
        case B::Max:
            return apply([] (Bits a, Bits b) { return SignedBits(b) > SignedBits(a) ? b : a; });
        case B::Atan2: break;
        }
    } else if (is_unsigned_integer_dtype(dtype)) {
        uint32_t width = basic_dtype_bitwidth(dtype);
        switch (op) {
        case B::Add: return apply([=] (Bits a, Bits b) { return unsigned_int_add(a, b, width); });
        case B::Sub: return apply([=] (Bits a, Bits b) { return unsigned_int_sub(a, b, width); });
        case B::Mul: return apply([=] (Bits a, Bits b) { return unsigned_int_mul(a, b, width); });
        case B::FloorDiv: return apply(unsigned_int_floordiv);
        case B::CDiv: return apply(unsigned_int_cdiv);
        case B::TrueDiv: break;
        case B::Mod: return apply(unsigned_int_mod);
        case B::Pow:
            return apply([=] (Bits a, Bits b) { return unsigned_int_pow(a, b, width); });
        case B::Min: return apply([] (Bits a, Bits b) { return b < a ? b : a; });
        case B::Max: return apply([] (Bits a, Bits b) { return b > a ? b : a; });
        case B::Atan2: break;
        }
    } else if (is_float_dtype(dtype)) {
        // Compute in double precision and round the result to `dtype`. For a single operation,
        // this is equivalent to computing directly in `dtype`, since double has more than
        // twice the precision of any narrower float dtype.
        auto apply_float = [&] (auto&& func) {
            return apply([dtype, &func] (Bits a, Bits b) -> Expected<Bits> {
                Expected<double> r = func(decode_float(a, dtype), decode_float(b, dtype));
                if (!r.is_ok()) return std::move(r.error());
                Bits bits;
                DTypeConversionStatus s = encode_float(*r, dtype, &bits);
                if (s == DTypeConversionStatus::OK) return bits;
                return dtype_conversion_status_to_error(
                        s, {bitcast_double_to_u64(*r), k_float64}, dtype);
            });
        };
        switch (op) {
        case B::Add:
            return apply_float([] (double a, double b) -> Expected<double> { return a + b; });
        case B::Sub:
            return apply_float([] (double a, double b) -> Expected<double> { return a - b; });
        case B::Mul:
            return apply_float([] (double a, double b) -> Expected<double> { return a * b; });
        case B::TrueDiv: return apply_float(float_truediv);
        case B::FloorDiv: return apply_float(float_floordiv);
        case B::CDiv: return apply_float(float_cdiv);
        case B::Mod: return apply_float(float_mod);
        case B::Pow: return apply_float(float_pow);
        // FIXME: respect propagate_nan
        case B::Min: return apply_float([] (double a, double b) -> Expected<double>
                                        { return b < a ? b : a; });
        case B::Max: return apply_float([] (double a, double b) -> Expected<double>
                                        { return b > a ? b : a; });
        case B::Atan2:
            return apply_float([&] (double a, double b) { return float_atan2(a, b); });
        }
    }

    return raise(PyExc_AssertionError,
                 "Unexpected constant binary arithmetic operation for dtype '", dtype, "'");
}

// Computes `x op y` elementwise modulo 2**bitwidth. Both tensors must be constant,
// with the same integer dtype and shape.
static Expected<RefPtr<TensorBase>> constant_tensor_wrapping_arith(TensorBase* x, TensorBase* y,
                                                                   BinaryWrappingOp op) {
    using W = BinaryWrappingOp;
    DType dtype = x->dtype;
    CHECK(y->dtype == dtype);
    uint32_t width = basic_dtype_bitwidth(dtype);

    auto apply = [&] (auto&& func) {
        return tensor_elementwise_apply_binary(x, y, dtype, func);
    };

    if (is_signed_integer_dtype(dtype)) {
        switch (op) {
        case W::Add:
            return apply([=] (Bits a, Bits b) { return truncate_int_to_signed(a + b, width); });
        case W::Sub:
            return apply([=] (Bits a, Bits b) { return truncate_int_to_signed(a - b, width); });
        case W::Mul:
            return apply([=] (Bits a, Bits b) { return truncate_int_to_signed(a * b, width); });
        }
    } else if (is_unsigned_integer_dtype(dtype)) {
        switch (op) {
        case W::Add:
            return apply([=] (Bits a, Bits b) { return truncate_int_to_unsigned(a + b, width); });
        case W::Sub:
            return apply([=] (Bits a, Bits b) { return truncate_int_to_unsigned(a - b, width); });
        case W::Mul:
            return apply([=] (Bits a, Bits b) { return truncate_int_to_unsigned(a * b, width); });
        }
    }

    return raise(PyExc_AssertionError,
                 "Unexpected constant wrapping arithmetic operation for dtype '", dtype, "'");
}

static bool prepare_tensorlike_operands(BorrowedValue x, BorrowedValue y,
                                        RefPtr<TensorBase>* x_tensor,
                                        RefPtr<TensorBase>* y_tensor) {
    if (x.is_tensor())
        *x_tensor = newref(x.as_tensor());
    else if (!x.is_loosely_typed_scalar())
        return false;

    if (y.is_tensor())
        *y_tensor = newref(y.as_tensor());
    else if (!y.is_loosely_typed_scalar())
        return false;

    return true;
}

static Expected<RefPtr<TensorBase>> fpowi_tensorlike(BorrowedValue x, BorrowedValue y,
                                                     const RefPtr<TensorBase>& x_tensor,
                                                     const RefPtr<TensorBase>& y_tensor) {
    bool use_tile = (x_tensor ? x_tensor : y_tensor)->kind == TensorBase::kTile;
    RefPtr<TensorBase> base = x_tensor, exponent = y_tensor;
    if (!base) {
        Expected<RefPtr<TensorBase>> r = loosely_typed_scalar_to_default_dtype(x, use_tile);
        if (!r.is_ok()) return std::move(r.error());
        base = std::move(*r);
    }
    if (!exponent) {
        Expected<RefPtr<TensorBase>> r = loosely_typed_scalar_to_default_dtype(y, use_tile);
        if (!r.is_ok()) return std::move(r.error());
        exponent = std::move(*r);
    }

    Error err = broadcast_operands(&base, &exponent);
    if (!err.is_ok()) return std::move(err);

    PyPtr base_var = tensor_to_ir_var(base.get());
    if (!base_var) return ErrorRaised;
    PyPtr exp_var = tensor_to_ir_var(exponent.get());
    if (!exp_var) return ErrorRaised;

    static LazyImporter fpowi_func;
    PyObject* func = fpowi_func.get("cuda.tile._ir.arithmetic_ops", "fpowi");
    if (!func) return ErrorRaised;
    PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
            func, base_var.get(), exp_var.get(), nullptr));
    if (!result_var) return ErrorRaised;

    RefPtr<TensorBase> res = tensor_symbolic_like(base.get(), base->dtype,
                                                  std::move(result_var));
    if (!res) return ErrorRaised;
    return res;
}

static RefPtr<TensorBase> symbolic_tensor_arith(TensorBase* x, TensorBase* y,
                                                BinaryArithmeticOp op,
                                                std::optional<RoundingMode> rounding_mode,
                                                bool flush_to_zero, bool propagate_nan) {
    using B = BinaryArithmeticOp;
    PyPtr x_var = tensor_to_ir_var(x);
    if (!x_var) return {};
    PyPtr y_var = tensor_to_ir_var(y);
    if (!y_var) return {};

    PyPtr result_var;
    if (op == B::Mod) {
        // Python-style modulo needs a sign correction on top of the C-style remainder
        static LazyImporter mod_func;
        PyObject* func = mod_func.get("cuda.tile._ir.arithmetic_ops", "mod_tensorlike_raw");
        if (!func) return {};
        result_var = steal(PyObject_CallFunctionObjArgs(func, x_var.get(), y_var.get(), nullptr));
    } else {
        PyObject* fn = nullptr;
        switch (op) {
        case B::Add: fn = g_add_pyunicode; break;
        case B::Sub: fn = g_sub_pyunicode; break;
        case B::Mul: fn = g_mul_pyunicode; break;
        case B::TrueDiv: fn = g_truediv_pyunicode; break;
        case B::FloorDiv: fn = g_floordiv_pyunicode; break;
        case B::CDiv: fn = g_cdiv_pyunicode; break;
        case B::Pow: fn = g_pow_pyunicode; break;
        case B::Min: fn = g_min_pyunicode; break;
        case B::Max: fn = g_max_pyunicode; break;
        case B::Atan2: fn = g_atan2_pyunicode; break;
        case B::Mod: CHECK_UNREACHABLE;
        }

        static LazyImporter raw_func;
        PyObject* func = raw_func.get(
                "cuda.tile._ir.arithmetic_ops", "binary_arithmetic_tensorlike_raw");
        if (!func) return {};

        PyObject* py_rounding_mode = rounding_mode.has_value()
                ? enum_to_pyobject(*rounding_mode) : Py_None;
        PyObject* py_ftz = flush_to_zero ? Py_True : Py_False;
        PyObject* py_propnan = propagate_nan ? Py_True : Py_False;

        result_var = steal(PyObject_CallFunctionObjArgs(
                func, fn, x_var.get(), y_var.get(), py_rounding_mode, py_ftz, py_propnan, nullptr));
    }
    if (!result_var) return {};

    return tensor_symbolic_like(x, x->dtype, std::move(result_var));
}

static Expected<Value> binary_arithmetic_tensorlike(
        BorrowedValue x, BorrowedValue y, BinaryArithmeticOp op,
        std::optional<RoundingMode> rounding_mode = {},
        bool flush_to_zero = false,
        bool propagate_nan = false) {
    RefPtr<TensorBase> x_tensor, y_tensor;
    if (!prepare_tensorlike_operands(x, y, &x_tensor, &y_tensor))
        return Value::not_implemented();

    using B = BinaryArithmeticOp;
    if (!x_tensor && !y_tensor) {
        PyObject* x_obj = x.as_loosely_typed_scalar();
        PyObject* y_obj = y.as_loosely_typed_scalar();
        PyObject* res = nullptr;
        switch (op) {
        case B::Add: res = PyNumber_Add(x_obj, y_obj); break;
        case B::Sub: res = PyNumber_Subtract(x_obj, y_obj); break;
        case B::Mul: res = PyNumber_Multiply(x_obj, y_obj); break;
        case B::TrueDiv: res = PyNumber_TrueDivide(x_obj, y_obj); break;
        case B::FloorDiv: res = PyNumber_FloorDivide(x_obj, y_obj); break;
        case B::CDiv: res = pynumber_cdiv(x_obj, y_obj).release(); break;
        case B::Mod: res = PyNumber_Remainder(x_obj, y_obj); break;
        case B::Pow: res = PyNumber_Power(x_obj, y_obj, Py_None); break;
        case B::Min: res = pynumber_min_max(x_obj, y_obj, false).release(); break;
        case B::Max: res = pynumber_min_max(x_obj, y_obj, true).release(); break;
        case B::Atan2: res = pynumber_atan2(x_obj, y_obj).release(); break;
        }
        if (!res) return error_from_python_exception();
        PyPtr res_guard = steal(res);
        BorrowedValue bv = borrowed_value_from_pyobject(res);
        if (bv.is_unsupported_pyobject())
            return raise(PyExc_AssertionError, "Arithmetic operation on Python numbers"
                         " returned an object of unexpected type ", Py_TYPE(res)->tp_name);
        Value val = bv.steal();
        res_guard.release();
        return val;
    }

    // Handle non-numeric cases (e.g. pointer addition)
    bool x_numeric = !x_tensor || is_numeric_dtype(x_tensor->dtype);
    bool y_numeric = !y_tensor || is_numeric_dtype(y_tensor->dtype);
    if (!x_numeric || !y_numeric) {
        bool x_pointer = x_tensor && is_pointer_dtype(x_tensor->dtype);
        bool y_pointer = y_tensor && is_pointer_dtype(y_tensor->dtype);
        if (x_pointer && y_numeric && (op == B::Add || op == B::Sub)) {
            if (!y_tensor) {
                Expected<RefPtr<TensorBase>> r = loosely_typed_scalar_to_default_dtype(
                        y, x_tensor->kind == TensorBase::kTile);
                if (!r.is_ok()) return std::move(r.error());
                y_tensor = std::move(*r);
            }
            return pointer_add_sub_offset(x_tensor.get(), y_tensor.get(), op);
        } else if (y_pointer && x_numeric && op == B::Add) {
            if (!x_tensor) {
                Expected<RefPtr<TensorBase>> r = loosely_typed_scalar_to_default_dtype(
                        x, y_tensor->kind == TensorBase::kTile);
                if (!r.is_ok()) return std::move(r.error());
                x_tensor = std::move(*r);
            }
            return pointer_add_sub_offset(y_tensor.get(), x_tensor.get(), op);
        } else {
            return Value::not_implemented();
        }
    }

    // Restricted float dtypes (e.g. float8) only support conversions, not arithmetic
    if (x_tensor && !is_arithmetic_numeric_dtype(x_tensor->dtype))
        return make_error(Error::kTypeError,
                          "Left-hand side must have an arithmetic dtype, not ", x_tensor->dtype);
    if (y_tensor && !is_arithmetic_numeric_dtype(y_tensor->dtype))
        return make_error(Error::kTypeError,
                          "Right-hand side must have an arithmetic dtype, not ", y_tensor->dtype);

    // For non-constant float**int, dispatch to specialized `fpowi` implementation,
    // which doesn't support rounding modes
    if (op == B::Pow && !rounding_mode.has_value()
            && ((x_tensor && x_tensor->is_symbolic()) || (y_tensor && y_tensor->is_symbolic()))
            && x.is_unrestricted_float_tensor_like() && y.is_integer_tensor_like()) {
        Expected<RefPtr<TensorBase>> res = fpowi_tensorlike(x, y, x_tensor, y_tensor);
        if (!res.is_ok()) return std::move(res.error());
        return Value::from_tensor(std::move(*res));
    }

    bool force_float = op == B::TrueDiv;
    Error promoted = promote_operands(&x_tensor, &y_tensor, x, y, force_float);
    if (!promoted.is_ok()) return std::move(promoted);

    if (op == B::Atan2 && !is_float_dtype(x_tensor->dtype))
        return make_error(Error::kTypeError,
                          "Operands of atan2 must have a float dtype, not ", x_tensor->dtype);

    if (x_tensor->dtype == k_bool_)
        return make_error(Error::kTypeError,
                          "Binary arithmetic is not supported for booleans."
                          " Please cast to an integer explicitly.");

    bool symbolic = x_tensor->is_symbolic() || y_tensor->is_symbolic();
    if (op == B::Pow && symbolic && is_integer_dtype(x_tensor->dtype))
        return make_error(Error::kTypeError, "Exponentiation of non-constant integers is not"
                          " supported. Please cast the base to a float dtype.");

    Error broadcasted = broadcast_operands(&x_tensor, &y_tensor);
    if (!broadcasted.is_ok()) return std::move(broadcasted);

    // At this point, both tensors have same dtype & shape
    if (!symbolic) {
        Expected<RefPtr<TensorBase>> res = constant_tensor_arith(
                x_tensor.get(), y_tensor.get(), op);
        if (!res.is_ok()) return std::move(res.error());
        return Value::from_tensor(std::move(*res));
    }

    RefPtr<TensorBase> res = symbolic_tensor_arith(
            x_tensor.get(), y_tensor.get(), op, rounding_mode, flush_to_zero, propagate_nan);
    if (!res) return ErrorRaised;
    return Value::from_tensor(std::move(res));
}

static std::optional<BinaryArithmeticOp> binary_arithmetic_op_from_fn(PyObject* str) {
    PyPtr interned = newref(str);
    pyunicode_intern_in_place(&interned);

    #define BINARY_OP_FROM_FN_CASE(name, fn, _op) \
        if (interned.get() == g_##fn##_pyunicode) return BinaryArithmeticOp::name;
    FOREACH_BINARY_ARITHMETIC_OP(BINARY_OP_FROM_FN_CASE);
    #undef BINARY_OP_FROM_FN_CASE

    return std::nullopt;
}

static PyObject* py_binary_arithmetic_tensorlike(PyObject*, PyObject* args) {
    PyObject *fn, *x, *y, *py_rounding_mode;
    int flush_to_zero, propagate_nan;
    if (!PyArg_ParseTuple(args, "UOOOpp:_binary_arithmetic_tensorlike",
            &fn, &x, &y, &py_rounding_mode, &flush_to_zero, &propagate_nan))
        return nullptr;

    std::optional<BinaryArithmeticOp> op = binary_arithmetic_op_from_fn(fn);
    if (!op.has_value()) {
        raise(PyExc_ValueError, "Unknown binary arithmetic operation ", use_repr(fn));
        return nullptr;
    }

    std::optional<RoundingMode> rounding_mode;
    if (!optional_rounding_mode_from_pyobject(py_rounding_mode, &rounding_mode))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    BorrowedValue y_val = borrowed_value_from_pyobject(y);
    Expected<Value> res = binary_arithmetic_tensorlike(
            x_val, y_val, *op, rounding_mode, flush_to_zero, propagate_nan);

    if (res.is_ok() && res->is_not_implemented()) {
        res = make_error(Error::kTypeError, "Unsupported operand types for ",
                binary_arithmetic_op_str(*op), ": '", use_type(x_val), "' and '",
                use_type(y_val), "'");
    }
    if (!res.is_ok()) {
        res.error().raise_compile_time_error();
        return nullptr;
    }
    return res->borrow().to_pyobject().release();
}


static RefPtr<TensorBase> truncate_pylong_to_tensor(PyObject* pylong, DType to, bool use_tile) {
    Result<Bits> r = truncate_pylong(pylong, to);
    if (!r.is_ok()) return {};
    return tensor_new_scalar_or_0d_tile(to, *r, use_tile);
}

// Unlike binary_arithmetic_tensorlike(), doesn't promote the operands, because the result
// depends on the bit width. A loosely typed operand is converted to the other operand's dtype,
// modulo 2**bitwidth.
static Expected<Value> binary_wrapping_arithmetic_tensorlike(
        BorrowedValue x, BorrowedValue y, BinaryWrappingOp op) {
    if (!x.is_integer_tensor_like())
        return make_error(Error::kTypeError, "Left-hand side of ", binary_wrapping_op_str(op),
                          " must be an integer, not ", use_type(x));
    if (!y.is_integer_tensor_like())
        return make_error(Error::kTypeError, "Right-hand side of ", binary_wrapping_op_str(op),
                          " must be an integer, not ", use_type(y));

    RefPtr<TensorBase> x_tensor, y_tensor;
    bool prepared = prepare_tensorlike_operands(x, y, &x_tensor, &y_tensor);
    CHECK(prepared);

    if (!x_tensor && !y_tensor) {
        return make_error(Error::kTypeError, binary_wrapping_op_str(op),
                          " needs an explicit bit width, so its operands cannot both be"
                          " loosely typed constants. Convert one of them to a typed"
                          " integer first, e.g. int32(x)");
    } else if (!x_tensor) {
        x_tensor = truncate_pylong_to_tensor(
                x.as_loosely_typed_scalar(), y_tensor->dtype, y_tensor->kind == TensorBase::kTile);
        if (!x_tensor) return ErrorRaised;
    } else if (!y_tensor) {
        y_tensor = truncate_pylong_to_tensor(
                y.as_loosely_typed_scalar(), x_tensor->dtype, x_tensor->kind == TensorBase::kTile);
        if (!y_tensor) return ErrorRaised;
    } else if (x_tensor->dtype != y_tensor->dtype) {
        return make_error(Error::kTypeError, "Operands of ", binary_wrapping_op_str(op),
                          " must have the same dtype, not ", x_tensor->dtype, " and ",
                          y_tensor->dtype);
    }

    Error broadcasted = broadcast_operands(&x_tensor, &y_tensor);
    if (!broadcasted.is_ok()) return std::move(broadcasted);

    if (!x_tensor->is_symbolic() && !y_tensor->is_symbolic()) {
        Expected<RefPtr<TensorBase>> res = constant_tensor_wrapping_arith(
                x_tensor.get(), y_tensor.get(), op);
        if (!res.is_ok()) return std::move(res.error());
        return Value::from_tensor(std::move(*res));
    }

    // FIXME: dispatch to a specialized implementation
    BinaryArithmeticOp arith_op = BinaryArithmeticOp::Add;
    switch (op) {
    case BinaryWrappingOp::Add: arith_op = BinaryArithmeticOp::Add; break;
    case BinaryWrappingOp::Sub: arith_op = BinaryArithmeticOp::Sub; break;
    case BinaryWrappingOp::Mul: arith_op = BinaryArithmeticOp::Mul; break;
    }

    RefPtr<TensorBase> res = symbolic_tensor_arith(
            x_tensor.get(), y_tensor.get(), arith_op, std::nullopt, false, false);
    if (!res) return ErrorRaised;
    return Value::from_tensor(std::move(res));
}

static std::optional<BinaryWrappingOp> binary_wrapping_op_from_fn(PyObject* str) {
    PyPtr interned = newref(str);
    pyunicode_intern_in_place(&interned);

    #define WRAPPING_OP_FROM_FN_CASE(name, fn) \
        if (interned.get() == g_##fn##_pyunicode) return BinaryWrappingOp::name;
    FOREACH_BINARY_WRAPPING_OP(WRAPPING_OP_FROM_FN_CASE);
    #undef WRAPPING_OP_FROM_FN_CASE

    return std::nullopt;
}

static PyObject* py_binary_wrapping_arithmetic_tensorlike(PyObject*, PyObject* args) {
    PyObject *fn, *x, *y;
    int compile_time_errors;
    if (!PyArg_ParseTuple(args, "UOOp:_binary_wrapping_arithmetic_tensorlike",
                          &fn, &x, &y, &compile_time_errors))
        return nullptr;

    std::optional<BinaryWrappingOp> op = binary_wrapping_op_from_fn(fn);
    if (!op.has_value()) {
        raise(PyExc_ValueError, "Unknown binary wrapping arithmetic operation ", use_repr(fn));
        return nullptr;
    }

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    BorrowedValue y_val = borrowed_value_from_pyobject(y);
    return release_pyobject_or_raise(
            binary_wrapping_arithmetic_tensorlike(x_val, y_val, *op), compile_time_errors);
}


static Expected<RefPtr<TensorBase>> constant_tensor_compare(TensorBase* x, TensorBase* y,
                                                            ComparisonOp op) {
    using C = ComparisonOp;
    DType dtype = x->dtype;
    CHECK(y->dtype == dtype);

    auto apply = [&] (auto&& func) {
        return tensor_elementwise_apply_binary(x, y, k_bool_, func);
    };

    if (is_signed_integer_dtype(dtype)) {
        using S = SignedBits;
        switch (op) {
        case C::Eq: return apply([] (Bits a, Bits b) { return a == b; });
        case C::Ne: return apply([] (Bits a, Bits b) { return a != b; });
        case C::Lt: return apply([] (Bits a, Bits b) { return S(a) < S(b); });
        case C::Le: return apply([] (Bits a, Bits b) { return S(a) <= S(b); });
        case C::Gt: return apply([] (Bits a, Bits b) { return S(a) > S(b); });
        case C::Ge: return apply([] (Bits a, Bits b) { return S(a) >= S(b); });
        }
    } else if (is_unsigned_integer_dtype(dtype) || dtype == k_bool_) {
        switch (op) {
        case C::Eq: return apply([] (Bits a, Bits b) { return a == b; });
        case C::Ne: return apply([] (Bits a, Bits b) { return a != b; });
        case C::Lt: return apply([] (Bits a, Bits b) { return a < b; });
        case C::Le: return apply([] (Bits a, Bits b) { return a <= b; });
        case C::Gt: return apply([] (Bits a, Bits b) { return a > b; });
        case C::Ge: return apply([] (Bits a, Bits b) { return a >= b; });
        }
    } else if (is_float_dtype(dtype)) {
        // Comparisons involving NaN are false, except for `!=`, as in Python.
        auto d = [dtype] (Bits bits) { return decode_float(bits, dtype); };
        switch (op) {
        case C::Eq: return apply([d] (Bits a, Bits b) { return d(a) == d(b); });
        case C::Ne: return apply([d] (Bits a, Bits b) { return d(a) != d(b); });
        case C::Lt: return apply([d] (Bits a, Bits b) { return d(a) < d(b); });
        case C::Le: return apply([d] (Bits a, Bits b) { return d(a) <= d(b); });
        case C::Gt: return apply([d] (Bits a, Bits b) { return d(a) > d(b); });
        case C::Ge: return apply([d] (Bits a, Bits b) { return d(a) >= d(b); });
        }
    }

    return raise(PyExc_AssertionError, "Unexpected constant comparison for dtype '", dtype, "'");
}

static Expected<Value> comparison_tensorlike(
        BorrowedValue x, BorrowedValue y, ComparisonOp op) {
    RefPtr<TensorBase> x_tensor, y_tensor;
    if (!prepare_tensorlike_operands(x, y, &x_tensor, &y_tensor))
        return Value::not_implemented();

    using C = ComparisonOp;
    if (!x_tensor && !y_tensor) {
        PyObject* res = PyObject_RichCompare(
                x.as_loosely_typed_scalar(), y.as_loosely_typed_scalar(), static_cast<int>(op));
        if (!res) return error_from_python_exception();
        PyPtr res_guard = steal(res);
        BorrowedValue bv = borrowed_value_from_pyobject(res);
        if (bv.is_unsupported_pyobject())
            return raise(PyExc_AssertionError, "Comparison of Python numbers"
                         " returned an object of unexpected type ", Py_TYPE(res)->tp_name);
        Value val = bv.steal();
        res_guard.release();
        return val;
    }

    // Non-numeric tensors (e.g. pointers) can only be compared with tensors of the same dtype
    bool x_numeric = !x_tensor || is_numeric_dtype(x_tensor->dtype);
    bool y_numeric = !y_tensor || is_numeric_dtype(y_tensor->dtype);
    if (!x_numeric || !y_numeric) {
        if (!x_tensor || !y_tensor || x_tensor->dtype != y_tensor->dtype) {
            if (!x_tensor)
                return make_error(Error::kTypeError, "Cannot compare a Python number"
                                  " with a value of dtype ", y_tensor->dtype);
            if (!y_tensor)
                return make_error(Error::kTypeError, "Cannot compare a value of dtype ",
                                  x_tensor->dtype, " with a Python number");
            return make_error(Error::kTypeError, "Cannot compare values of dtypes ",
                              x_tensor->dtype, " and ", y_tensor->dtype);
        }
    }

    // Tensors of the same non-numeric dtype pass through promotion unchanged
    Error promoted = promote_operands(&x_tensor, &y_tensor, x, y, false);
    if (!promoted.is_ok()) return std::move(promoted);
    Error broadcasted = broadcast_operands(&x_tensor, &y_tensor);
    if (!broadcasted.is_ok()) return std::move(broadcasted);

    // At this point, both tensors have same dtype & shape
    if (!x_tensor->is_symbolic() && !y_tensor->is_symbolic()) {
        Expected<RefPtr<TensorBase>> res = constant_tensor_compare(
                x_tensor.get(), y_tensor.get(), op);
        if (!res.is_ok()) return std::move(res.error());
        return Value::from_tensor(std::move(*res));
    }

    PyPtr x_var = tensor_to_ir_var(x_tensor.get());
    if (!x_var) return ErrorRaised;
    PyPtr y_var = tensor_to_ir_var(y_tensor.get());
    if (!y_var) return ErrorRaised;

    PyObject* fn = nullptr;
    switch (op) {
    #define COMPARISON_OP_FN_CASE(name, fn_name, _py_op, _op) \
        case C::name: fn = g_##fn_name##_pyunicode; break;
    FOREACH_COMPARISON_OP(COMPARISON_OP_FN_CASE);
    #undef COMPARISON_OP_FN_CASE
    }

    static LazyImporter compare_func;
    PyObject* func = compare_func.get(
            "cuda.tile._ir.arithmetic_ops", "compare_tensorlike_raw");
    if (!func) return ErrorRaised;

    PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
            func, fn, x_var.get(), y_var.get(), nullptr));
    if (!result_var) return ErrorRaised;

    RefPtr<TensorBase> res = tensor_symbolic_like(x_tensor.get(), k_bool_,
                                                  std::move(result_var));
    if (!res) return ErrorRaised;
    return Value::from_tensor(std::move(res));
}

static std::optional<ComparisonOp> comparison_op_from_fn(PyObject* str) {
    PyPtr interned = newref(str);
    pyunicode_intern_in_place(&interned);

    #define COMPARISON_OP_FROM_FN_CASE(name, fn, _py_op, _op) \
        if (interned.get() == g_##fn##_pyunicode) return ComparisonOp::name;
    FOREACH_COMPARISON_OP(COMPARISON_OP_FROM_FN_CASE);
    #undef COMPARISON_OP_FROM_FN_CASE

    return std::nullopt;
}

static PyObject* py_comparison_tensorlike(PyObject*, PyObject* args) {
    PyObject *fn, *x, *y;
    if (!PyArg_ParseTuple(args, "UOO:_comparison_tensorlike", &fn, &x, &y))
        return nullptr;

    std::optional<ComparisonOp> op = comparison_op_from_fn(fn);
    if (!op.has_value()) {
        raise(PyExc_ValueError, "Unknown comparison operation ", use_repr(fn));
        return nullptr;
    }

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    BorrowedValue y_val = borrowed_value_from_pyobject(y);
    Expected<Value> res = comparison_tensorlike(x_val, y_val, *op);

    if (res.is_ok() && res->is_not_implemented()) {
        res = make_error(Error::kTypeError, "Unsupported operand types for ",
                comparison_op_str(*op), ": '", use_type(x_val), "' and '",
                use_type(y_val), "'");
    }
    return release_pyobject_or_raise(std::move(res), true);
}

static PyObject* tensor_richcompare(PyObject* self, PyObject* other, int py_op) {
    CHECK(py_op >= Py_LT && py_op <= Py_GE);
    ComparisonOp op = static_cast<ComparisonOp>(py_op);

    BorrowedValue xval = borrowed_value_from_pyobject(self);
    BorrowedValue yval = borrowed_value_from_pyobject(other);
    Result<Value> res = to_result(comparison_tensorlike(xval, yval, op));
    if (!res.is_ok()) return nullptr;

    return res->borrow().to_pyobject().release();
}

// Computes `x op y` elementwise. Both tensors must be constant, with the same dtype and shape.
static Expected<RefPtr<TensorBase>> constant_tensor_binary_bitwise(
        TensorBase* x, TensorBase* y, BinaryBitwiseOp op) {
    using B = BinaryBitwiseOp;
    DType dtype = x->dtype;
    CHECK(y->dtype == dtype);

    auto apply = [&] (auto&& func) {
        return tensor_elementwise_apply_binary(x, y, dtype, func);
    };

    // Signed integers are stored sign-extended and unsigned ones zero-extended,
    // and bitwise operations preserve both, so no truncation is needed.
    switch (op) {
    case B::And: return apply([] (Bits a, Bits b) { return a & b; });
    case B::Or: return apply([] (Bits a, Bits b) { return a | b; });
    case B::Xor: return apply([] (Bits a, Bits b) { return a ^ b; });
    }
    CHECK_UNREACHABLE;
}

static Expected<Value> binary_bitwise_tensorlike(
        BorrowedValue x, BorrowedValue y, BinaryBitwiseOp op) {
    RefPtr<TensorBase> x_tensor, y_tensor;
    if (!prepare_tensorlike_operands(x, y, &x_tensor, &y_tensor))
        return Value::not_implemented();

    using B = BinaryBitwiseOp;
    if (!x_tensor && !y_tensor) {
        PyObject* x_obj = x.as_loosely_typed_scalar();
        PyObject* y_obj = y.as_loosely_typed_scalar();
        PyObject* res = nullptr;
        switch (op) {
        case B::And: res = PyNumber_And(x_obj, y_obj); break;
        case B::Or: res = PyNumber_Or(x_obj, y_obj); break;
        case B::Xor: res = PyNumber_Xor(x_obj, y_obj); break;
        }
        if (!res) return error_from_python_exception();
        PyPtr res_guard = steal(res);
        BorrowedValue bv = borrowed_value_from_pyobject(res);
        if (bv.is_unsupported_pyobject())
            return raise(PyExc_AssertionError, "Bitwise operation on Python numbers"
                         " returned an object of unexpected type ", Py_TYPE(res)->tp_name);
        Value val = bv.steal();
        res_guard.release();
        return val;
    }

    if (!(x.is_integer_tensor_like() || x.is_boolean_tensor_like())
            || !(y.is_integer_tensor_like() || y.is_boolean_tensor_like()))
        return make_error(Error::kTypeError, "Bitwise operations require integers or booleans."
                          " Use an explicit cuda.tile.bitcast() for non-integer operands.");

    if (x_tensor && y_tensor && x_tensor->dtype != y_tensor->dtype)
        return make_error(Error::kTypeError, "Bitwise operands must have the same dtype, not ",
                          x_tensor->dtype, " and ", y_tensor->dtype);

    // A loosely typed integer is never int8, so this can only be an int8 tensor and a Python bool
    if ((x_tensor && x_tensor->dtype == k_int8 && y.is_loosely_typed_bool())
            || (y_tensor && y_tensor->dtype == k_int8 && x.is_loosely_typed_bool()))
        return make_error(Error::kTypeError,
                          "Bitwise operations do not support bool and int8 operands");

    Error promoted = promote_operands(&x_tensor, &y_tensor, x, y, false);
    if (!promoted.is_ok()) return std::move(promoted);
    Error broadcasted = broadcast_operands(&x_tensor, &y_tensor);
    if (!broadcasted.is_ok()) return std::move(broadcasted);

    // At this point, both tensors have same dtype & shape
    DType dtype = x_tensor->dtype;
    if (!x_tensor->is_symbolic() && !y_tensor->is_symbolic()) {
        Expected<RefPtr<TensorBase>> res = constant_tensor_binary_bitwise(
                x_tensor.get(), y_tensor.get(), op);
        if (!res.is_ok()) return std::move(res.error());
        return Value::from_tensor(std::move(*res));
    }

    PyPtr x_var = tensor_to_ir_var(x_tensor.get());
    if (!x_var) return ErrorRaised;
    PyPtr y_var = tensor_to_ir_var(y_tensor.get());
    if (!y_var) return ErrorRaised;

    PyObject* fn = nullptr;
    switch (op) {
    #define BITWISE_OP_FN_CASE(name, fn_name, _op) \
        case B::name: fn = g_##fn_name##_pyunicode; break;
    FOREACH_BINARY_BITWISE_OP(BITWISE_OP_FN_CASE);
    #undef BITWISE_OP_FN_CASE
    }

    static LazyImporter bitwise_func;
    PyObject* func = bitwise_func.get(
            "cuda.tile._ir.arithmetic_ops", "binary_bitwise_tensorlike_raw");
    if (!func) return ErrorRaised;

    PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
            func, fn, x_var.get(), y_var.get(), nullptr));
    if (!result_var) return ErrorRaised;

    RefPtr<TensorBase> res = tensor_symbolic_like(x_tensor.get(), dtype,
                                                  std::move(result_var));
    if (!res) return ErrorRaised;
    return Value::from_tensor(std::move(res));
}

static std::optional<BinaryBitwiseOp> binary_bitwise_op_from_fn(PyObject* str) {
    PyPtr interned = newref(str);
    pyunicode_intern_in_place(&interned);

    #define BITWISE_OP_FROM_FN_CASE(name, fn, _op) \
        if (interned.get() == g_##fn##_pyunicode) return BinaryBitwiseOp::name;
    FOREACH_BINARY_BITWISE_OP(BITWISE_OP_FROM_FN_CASE);
    #undef BITWISE_OP_FROM_FN_CASE

    return std::nullopt;
}

static PyObject* py_binary_bitwise_tensorlike(PyObject*, PyObject* args) {
    PyObject *fn, *x, *y;
    if (!PyArg_ParseTuple(args, "UOO:_binary_bitwise_tensorlike", &fn, &x, &y))
        return nullptr;

    std::optional<BinaryBitwiseOp> op = binary_bitwise_op_from_fn(fn);
    if (!op.has_value()) {
        raise(PyExc_ValueError, "Unknown binary bitwise operation ", use_repr(fn));
        return nullptr;
    }

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    BorrowedValue y_val = borrowed_value_from_pyobject(y);
    Expected<Value> res = binary_bitwise_tensorlike(x_val, y_val, *op);

    if (res.is_ok() && res->is_not_implemented()) {
        res = make_error(Error::kTypeError, "Unsupported operand types for ",
                binary_bitwise_op_str(*op), ": '", use_type(x_val), "' and '",
                use_type(y_val), "'");
    }
    return release_pyobject_or_raise(std::move(res), true);
}

static Error shift_count_out_of_range_error(Bits count, DType dtype) {
    return make_error(Error::kValueError, "Shift count ", count, " is out of range for ", dtype);
}

// Computes `x op y` elementwise. Both tensors must be constant, with the same integer dtype and
// shape. The shift count must be non-negative and less than the dtype's width.
// For left shifts, the bits shifted out of the dtype's width are discarded.
static Expected<RefPtr<TensorBase>> constant_tensor_bitwise_shift(
        TensorBase* x, TensorBase* y, BitwiseShiftOp op) {
    using S = BitwiseShiftOp;
    DType dtype = x->dtype;
    CHECK(y->dtype == dtype);
    CHECK(is_integer_dtype(dtype));
    uint32_t width = basic_dtype_bitwidth(dtype);

    auto apply = [&] (auto&& func) {
        return tensor_elementwise_apply_binary(x, y, dtype, func);
    };

    if (is_signed_integer_dtype(dtype)) {
        switch (op) {
        case S::LShift:
            return apply([width, dtype] (Bits a, Bits b) -> Expected<Bits> {
                if (SignedBits(b) < 0)
                    return make_error(Error::kValueError, "Negative shift count");
                if (b >= width) return shift_count_out_of_range_error(b, dtype);
                return truncate_int_to_signed(a << b, width);
            });
        case S::RShift:
            return apply([width, dtype] (Bits a, Bits b) -> Expected<Bits> {
                if (SignedBits(b) < 0)
                    return make_error(Error::kValueError, "Negative shift count");
                if (b >= width) return shift_count_out_of_range_error(b, dtype);
                return Bits(SignedBits(a) >> b);
            });
        }
    } else {
        switch (op) {
        case S::LShift:
            return apply([width, dtype] (Bits a, Bits b) -> Expected<Bits> {
                if (b >= width) return shift_count_out_of_range_error(b, dtype);
                return truncate_int_to_unsigned(a << b, width);
            });
        case S::RShift:
            return apply([width, dtype] (Bits a, Bits b) -> Expected<Bits> {
                if (b >= width) return shift_count_out_of_range_error(b, dtype);
                return a >> b;
            });
        }
    }
    CHECK_UNREACHABLE;
}

static Expected<Value> bitwise_shift_tensorlike(
        BorrowedValue x, BorrowedValue y, BitwiseShiftOp op) {
    RefPtr<TensorBase> x_tensor, y_tensor;
    if (!prepare_tensorlike_operands(x, y, &x_tensor, &y_tensor))
        return Value::not_implemented();

    using S = BitwiseShiftOp;
    if (!x_tensor && !y_tensor) {
        PyObject* x_obj = x.as_loosely_typed_scalar();
        PyObject* y_obj = y.as_loosely_typed_scalar();
        PyObject* res = nullptr;
        switch (op) {
        case S::LShift: res = PyNumber_Lshift(x_obj, y_obj); break;
        case S::RShift: res = PyNumber_Rshift(x_obj, y_obj); break;
        }
        if (!res) return error_from_python_exception();
        PyPtr res_guard = steal(res);
        BorrowedValue bv = borrowed_value_from_pyobject(res);
        if (bv.is_unsupported_pyobject())
            return raise(PyExc_AssertionError, "Bitwise shift on Python numbers"
                         " returned an object of unexpected type ", Py_TYPE(res)->tp_name);
        Value val = bv.steal();
        res_guard.release();
        return val;
    }

    // Booleans are not accepted, unlike in the other bitwise operations
    if (!x.is_integer_tensor_like())
        return make_error(Error::kTypeError,
                          "Left-hand side of bitwise shift must be an integer, not ", use_type(x));
    if (!y.is_integer_tensor_like())
        return make_error(Error::kTypeError,
                          "Right-hand side of bitwise shift must be an integer, not ", use_type(y));

    Error promoted = promote_operands(&x_tensor, &y_tensor, x, y, false);
    if (!promoted.is_ok()) return std::move(promoted);
    Error broadcasted = broadcast_operands(&x_tensor, &y_tensor);
    if (!broadcasted.is_ok()) return std::move(broadcasted);

    // At this point, both tensors have same dtype & shape
    DType dtype = x_tensor->dtype;
    if (!x_tensor->is_symbolic() && !y_tensor->is_symbolic()) {
        Expected<RefPtr<TensorBase>> res = constant_tensor_bitwise_shift(
                x_tensor.get(), y_tensor.get(), op);
        if (!res.is_ok()) return std::move(res.error());
        return Value::from_tensor(std::move(*res));
    }

    PyPtr x_var = tensor_to_ir_var(x_tensor.get());
    if (!x_var) return ErrorRaised;
    PyPtr y_var = tensor_to_ir_var(y_tensor.get());
    if (!y_var) return ErrorRaised;

    PyObject* fn = nullptr;
    switch (op) {
    #define SHIFT_OP_FN_CASE(name, fn_name, _op) \
        case S::name: fn = g_##fn_name##_pyunicode; break;
    FOREACH_BITWISE_SHIFT_OP(SHIFT_OP_FN_CASE);
    #undef SHIFT_OP_FN_CASE
    }

    static LazyImporter shift_func;
    PyObject* func = shift_func.get(
            "cuda.tile._ir.arithmetic_ops", "bitwise_shift_tensorlike_raw");
    if (!func) return ErrorRaised;

    PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
            func, fn, x_var.get(), y_var.get(), nullptr));
    if (!result_var) return ErrorRaised;

    RefPtr<TensorBase> res = tensor_symbolic_like(x_tensor.get(), dtype,
                                                  std::move(result_var));
    if (!res) return ErrorRaised;
    return Value::from_tensor(std::move(res));
}

static std::optional<BitwiseShiftOp> bitwise_shift_op_from_fn(PyObject* str) {
    PyPtr interned = newref(str);
    pyunicode_intern_in_place(&interned);

    #define SHIFT_OP_FROM_FN_CASE(name, fn, _op) \
        if (interned.get() == g_##fn##_pyunicode) return BitwiseShiftOp::name;
    FOREACH_BITWISE_SHIFT_OP(SHIFT_OP_FROM_FN_CASE);
    #undef SHIFT_OP_FROM_FN_CASE

    return std::nullopt;
}

static PyObject* py_bitwise_shift_tensorlike(PyObject*, PyObject* args) {
    PyObject *fn, *x, *y;
    if (!PyArg_ParseTuple(args, "UOO:_bitwise_shift_tensorlike", &fn, &x, &y))
        return nullptr;

    std::optional<BitwiseShiftOp> op = bitwise_shift_op_from_fn(fn);
    if (!op.has_value()) {
        raise(PyExc_ValueError, "Unknown bitwise shift operation ", use_repr(fn));
        return nullptr;
    }

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    BorrowedValue y_val = borrowed_value_from_pyobject(y);
    Expected<Value> res = bitwise_shift_tensorlike(x_val, y_val, *op);

    if (res.is_ok() && res->is_not_implemented()) {
        res = make_error(Error::kTypeError, "Unsupported operand types for ",
                bitwise_shift_op_str(*op), ": '", use_type(x_val), "' and '",
                use_type(y_val), "'");
    }
    return release_pyobject_or_raise(std::move(res), true);
}

template <BinaryArithmeticOp Op>
static PyObject* tensor_arith_binop(PyObject* x, PyObject* y) {
    BorrowedValue xval = borrowed_value_from_pyobject(x);
    BorrowedValue yval = borrowed_value_from_pyobject(y);
    Result<Value> res = to_result(binary_arithmetic_tensorlike(xval, yval, Op));
    if (!res.is_ok()) return nullptr;

    return res->borrow().to_pyobject().release();
}

// Returns a tuple (x // y, x % y), or NotImplemented
static Expected<Value> divmod_tensorlike(BorrowedValue x, BorrowedValue y) {
    if (x.is_boolean_tensor_like() && y.is_boolean_tensor_like())
        return make_error(Error::kTypeError, "divmod() does not support booleans");

    if (x.is_loosely_typed_scalar() && y.is_loosely_typed_scalar()) {
        PyPtr res = steal(PyNumber_Divmod(x.as_loosely_typed_scalar(),
                                          y.as_loosely_typed_scalar()));
        if (!res) return error_from_python_exception();
        if (!PyTuple_Check(res.get()))
            return raise(PyExc_TypeError, "divmod() on two Python numbers was expected "
                         "to return a tuple, not ", use_repr(res.get()));
        return BorrowedValue::from_pytuple(res.release()).steal();
    }

    bool symbolic = (x.is_tensor() && x.as_tensor()->is_symbolic())
            || (y.is_tensor() && y.as_tensor()->is_symbolic());
    if (symbolic && (x.is_unrestricted_float_tensor_like()
                     || y.is_unrestricted_float_tensor_like()))
        return make_error(Error::kTypeError,
                          "divmod() is not implemented for floating point values");

    Expected<Value> quotient = binary_arithmetic_tensorlike(
            x, y, BinaryArithmeticOp::FloorDiv);
    if (!quotient.is_ok()) return std::move(quotient.error());
    if (quotient->is_not_implemented()) return BorrowedValue::not_implemented().to_owned();

    Expected<Value> remainder = binary_arithmetic_tensorlike(x, y, BinaryArithmeticOp::Mod);
    if (!remainder.is_ok()) return std::move(remainder.error());
    if (remainder->is_not_implemented()) return BorrowedValue::not_implemented().to_owned();

    return Value::pack_tuple(*std::move(quotient), *std::move(remainder));
}

static PyObject* tensor_nb_divmod(PyObject* x, PyObject* y) {
    BorrowedValue xval = borrowed_value_from_pyobject(x);
    BorrowedValue yval = borrowed_value_from_pyobject(y);
    Expected<Value> res = divmod_tensorlike(xval, yval);
    return release_pyobject_or_raise(std::move(res), false);
}

static PyObject* py_divmod_tensorlike(PyObject*, PyObject* args) {
    PyObject *x, *y;
    if (!PyArg_ParseTuple(args, "OO:_divmod_tensorlike", &x, &y))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    BorrowedValue y_val = borrowed_value_from_pyobject(y);
    Expected<Value> res = divmod_tensorlike(x_val, y_val);

    if (res.is_ok() && res->is_not_implemented()) {
        res = make_error(Error::kTypeError, "Unsupported operand types for divmod(): '",
                use_type(x_val), "' and '", use_type(y_val), "'");
    }
    return release_pyobject_or_raise(std::move(res), true);
}

static PyObject* tensor_nb_power(PyObject* x, PyObject* y, PyObject* mod) {
    if (mod != Py_None) Py_RETURN_NOTIMPLEMENTED;
    return tensor_arith_binop<BinaryArithmeticOp::Pow>(x, y);
}

template <BinaryBitwiseOp Op>
static PyObject* tensor_bitwise_binop(PyObject* x, PyObject* y) {
    BorrowedValue xval = borrowed_value_from_pyobject(x);
    BorrowedValue yval = borrowed_value_from_pyobject(y);
    Result<Value> res = to_result(binary_bitwise_tensorlike(xval, yval, Op));
    if (!res.is_ok()) return nullptr;
    return res->borrow().to_pyobject().release();
}

template <BitwiseShiftOp Op>
static PyObject* tensor_bitwise_shift_binop(PyObject* x, PyObject* y) {
    BorrowedValue xval = borrowed_value_from_pyobject(x);
    BorrowedValue yval = borrowed_value_from_pyobject(y);
    Result<Value> res = to_result(bitwise_shift_tensorlike(xval, yval, Op));
    if (!res.is_ok()) return nullptr;
    return res->borrow().to_pyobject().release();
}


// ----- Tensors: conversion to Python numbers -----

static const char* tensor_kind_name(TensorBase::Kind kind) {
    switch (kind) {
    case TensorBase::kTile: return "tile";
    case TensorBase::kScalar: return "scalar";
    case TensorBase::kVector: return "vector";
    case TensorBase::kPointer: return "pointer";
    case TensorBase::kOpaque: return "opaque value";
    }
    CHECK_UNREACHABLE;
}

// Returns the value of a tensor that is known to be constant and zero-dimensional
static Bits zero_dim_constant_bits_unchecked(const TensorBase* tensor) {
    switch (tensor->tag_) {
    case TensorBase::kBits: return tensor->data_.bits;
    case TensorBase::kBitsArray: return tensor->data_.bits_array[0];
    case TensorBase::kVar: break;
    }
    CHECK_UNREACHABLE;
}

// Returns the value of a constant zero-dimensional tile or scalar. `action` completes
// the error messages, e.g. "be used as index".
static Expected<Bits> zero_dim_constant_bits(TensorBase* tensor, const char* action) {
    switch (tensor->kind) {
    case TensorBase::kTile:
        {
            const PackedShape& shape = static_cast<Tile*>(tensor)->shape_;
            if (shape.ndim() != 0)
                return make_error(Error::kTypeError, "Tile of shape ", shape, " cannot ", action);
            break;
        }
    case TensorBase::kScalar:
        break;
    case TensorBase::kVector:
        return make_error(Error::kTypeError, "Vector cannot ", action);
    case TensorBase::kPointer:
        return make_error(Error::kTypeError, "Pointer cannot ", action);
    case TensorBase::kOpaque:
        return make_error(Error::kTypeError, "OpaqueValue cannot ", action);
    }

    if (!is_numeric_dtype(tensor->dtype))
        return make_error(Error::kTypeError, use_type(BorrowedValue::from_tensor(tensor)),
                          " cannot ", action);
    if (tensor->is_symbolic())
        return make_error(Error::kValueError, "Symbolic ", tensor_kind_name(tensor->kind),
                          " has no concrete value and thus cannot ", action);
    return zero_dim_constant_bits_unchecked(tensor);
}

static PyObject* tensor_index(PyObject* self) {
    TensorBase* tensor = reinterpret_cast<TensorBase*>(self);
    Result<Bits> bits = to_result(zero_dim_constant_bits(tensor, "be used as index"));
    if (!bits.is_ok()) return nullptr;

    if (!is_integer_dtype(tensor->dtype)) {
        raise(PyExc_TypeError, "Index must be an integer, not ",
              use_type(BorrowedValue::from_tensor(tensor)));
        return nullptr;
    }
    return typed_scalar_to_pyobject(*bits, tensor->dtype).release();
}

static PyObject* tensor_int(PyObject* self) {
    TensorBase* tensor = reinterpret_cast<TensorBase*>(self);
    Result<Bits> bits = to_result(zero_dim_constant_bits(tensor, "be converted to int"));
    if (!bits.is_ok()) return nullptr;

    PyPtr number = typed_scalar_to_pyobject(*bits, tensor->dtype);
    if (!number) return nullptr;
    return PyNumber_Long(number.get());
}

static PyObject* tensor_float(PyObject* self) {
    TensorBase* tensor = reinterpret_cast<TensorBase*>(self);
    Result<Bits> bits = to_result(zero_dim_constant_bits(tensor, "be converted to float"));
    if (!bits.is_ok()) return nullptr;

    PyPtr number = typed_scalar_to_pyobject(*bits, tensor->dtype);
    if (!number) return nullptr;
    return PyNumber_Float(number.get());
}

static int tensor_bool(PyObject* self) {
    TensorBase* tensor = reinterpret_cast<TensorBase*>(self);
    Result<Bits> bits = to_result(zero_dim_constant_bits(tensor, "be converted to boolean"));
    if (!bits.is_ok()) return -1;

    PyPtr number = typed_scalar_to_pyobject(*bits, tensor->dtype);
    if (!number) return -1;
    return PyObject_IsTrue(number.get());
}

// ----- Tensors: unary elementwise -----

static PyObject* get_unary_raw_func() {
    static LazyImporter func;
    return func.get("cuda.tile._ir.arithmetic_ops", "unary_raw");
}

// Negates a constant integer or float tensor elementwise. Like subtraction, integer negation
// raises OverflowError if the result doesn't fit the dtype.
static Expected<RefPtr<TensorBase>> constant_tensor_negative(TensorBase* x) {
    DType dtype = x->dtype;
    if (is_signed_integer_dtype(dtype)) {
        uint32_t width = basic_dtype_bitwidth(dtype);
        SignedBits min_value = kSignedBitsMin >> (MaxWidth - width);
        return tensor_elementwise_apply(x, dtype,
                [width, min_value] (Bits a) -> Expected<Bits> {
            if (SignedBits(a) == min_value) return int_overflow_error("negation", true, width);
            return 0 - a;
        });
    } else if (is_unsigned_integer_dtype(dtype)) {
        uint32_t width = basic_dtype_bitwidth(dtype);
        return tensor_elementwise_apply(x, dtype, [width] (Bits a) -> Expected<Bits> {
            if (a != 0) return int_overflow_error("negation", false, width);
            return 0;
        });
    } else {
        CHECK(is_unrestricted_float_dtype(dtype));
        return tensor_elementwise_apply(x, dtype, [dtype] (Bits a) -> Expected<Bits> {
            Bits ret;
            DTypeConversionStatus s = encode_float(-decode_float(a, dtype), dtype, &ret);
            if (s == DTypeConversionStatus::OK) return ret;
            return dtype_conversion_status_to_error(s, {a, dtype}, dtype);
        });
    }
}

static Expected<RefPtr<TensorBase>> tensor_negative(RefPtr<TensorBase> x) {
    if (x->dtype == k_bool_) {
        Expected<RefPtr<TensorBase>> converted = tensor_astype(x.get(), k_int32);
        if (!converted.is_ok()) return std::move(converted.error());
        x = std::move(*converted);
    } else if (!is_arithmetic_numeric_dtype(x->dtype)) {
        return make_error(Error::kTypeError,
                          "Operand of unary `-` must have an arithmetic dtype, not ", x->dtype);
    }

    if (!x->is_symbolic())
        return constant_tensor_negative(x.get());

    PyObject* func = get_unary_raw_func();
    if (!func) return ErrorRaised;

    PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
            func, g_neg_pyunicode, x->var(), nullptr));
    if (!result_var) return ErrorRaised;

    RefPtr<TensorBase> res = tensor_symbolic_like(x.get(), x->dtype, std::move(result_var));
    if (!res) return ErrorRaised;
    return res;
}

static PyObject* tensor_nb_negative(PyObject* self) {
    Result<RefPtr<TensorBase>> res = to_result(tensor_negative(
            newref(reinterpret_cast<TensorBase*>(self))));
    if (!res.is_ok()) return nullptr;
    return reinterpret_cast<PyObject*>(res->release());
}

static Expected<RefPtr<TensorBase>> tensor_positive(RefPtr<TensorBase> x) {
    if (x->dtype == k_bool_)
        return tensor_astype(x.get(), k_int32);
    if (!is_arithmetic_numeric_dtype(x->dtype))
        return make_error(Error::kTypeError,
                          "Operand of unary `+` must have an arithmetic dtype, not ", x->dtype);
    return x;
}

static PyObject* tensor_nb_positive(PyObject* self) {
    Result<RefPtr<TensorBase>> res = to_result(tensor_positive(
            newref(reinterpret_cast<TensorBase*>(self))));
    if (!res.is_ok()) return nullptr;
    return reinterpret_cast<PyObject*>(res->release());
}

static Expected<RefPtr<TensorBase>> constant_tensor_invert(TensorBase* x) {
    DType dtype = x->dtype;
    if (is_signed_integer_dtype(dtype)) {
        // Stays sign-extended
        return tensor_elementwise_apply(x, dtype, [] (Bits a) { return ~a; });
    } else {
        CHECK(is_unsigned_integer_dtype(dtype));
        uint32_t width = basic_dtype_bitwidth(dtype);
        return tensor_elementwise_apply(x, dtype,
                [width] (Bits a) { return truncate_int_to_unsigned(~a, width); });
    }
}

static Expected<RefPtr<TensorBase>> tensor_invert(TensorBase* x) {
    if (x->dtype == k_bool_) {
        if (!x->is_symbolic())
            return make_error(Error::kTypeError, "`~` is not supported for boolean constants;"
                              " use ct.bitwise_not() instead");
    } else if (!is_integer_dtype(x->dtype)) {
        return make_error(Error::kTypeError,
                          "Operand of `~` must be an integer or boolean, not ", x->dtype);
    }

    if (!x->is_symbolic())
        return constant_tensor_invert(x);

    PyObject* func = get_unary_raw_func();
    if (!func) return ErrorRaised;

    PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
            func, g_invert_pyunicode, x->var(), nullptr));
    if (!result_var) return ErrorRaised;

    RefPtr<TensorBase> res = tensor_symbolic_like(x, x->dtype, std::move(result_var));
    if (!res) return ErrorRaised;
    return res;
}

static PyObject* tensor_nb_invert(PyObject* self) {
    Result<RefPtr<TensorBase>> res = to_result(tensor_invert(
            reinterpret_cast<TensorBase*>(self)));
    if (!res.is_ok()) return nullptr;
    return reinterpret_cast<PyObject*>(res->release());
}

// Takes ownership of `res`, which is the result of a unary operation on a loosely typed scalar
static Expected<Value> loosely_typed_unary_result(PyObject* res) {
    if (!res) return error_from_python_exception();
    PyPtr res_guard = steal(res);
    BorrowedValue bv = borrowed_value_from_pyobject(res);
    if (bv.is_unsupported_pyobject())
        return raise(PyExc_AssertionError, "Unary operation on a loosely typed scalar"
                     " returned an object of unexpected type ", Py_TYPE(res)->tp_name);
    Value val = bv.steal();
    res_guard.release();
    return val;
}

static Expected<Value> tensorlike_negative(BorrowedValue x) {
    if (x.is_loosely_typed_scalar())
        return loosely_typed_unary_result(PyNumber_Negative(x.as_loosely_typed_scalar()));
    if (!x.is_tensor())
        return make_error(Error::kTypeError,
                          "Operand of unary `-` must be a tensor or a number, not ", use_type(x));
    Expected<RefPtr<TensorBase>> res = tensor_negative(newref(x.as_tensor()));
    if (!res.is_ok()) return std::move(res.error());
    return Value::from_tensor(std::move(*res));
}

static PyObject* py_tensorlike_negative(PyObject*, PyObject* args) {
    PyObject* x;
    int compile_time_errors;
    if (!PyArg_ParseTuple(args, "Op:_tensorlike_negative", &x, &compile_time_errors))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    return release_pyobject_or_raise(tensorlike_negative(x_val), compile_time_errors);
}

static Expected<Value> tensorlike_positive(BorrowedValue x) {
    if (x.is_loosely_typed_scalar())
        return loosely_typed_unary_result(PyNumber_Positive(x.as_loosely_typed_scalar()));
    if (!x.is_tensor())
        return make_error(Error::kTypeError,
                          "Operand of unary `+` must be a tensor or a number, not ", use_type(x));
    Expected<RefPtr<TensorBase>> res = tensor_positive(newref(x.as_tensor()));
    if (!res.is_ok()) return std::move(res.error());
    return Value::from_tensor(std::move(*res));
}

static PyObject* py_tensorlike_positive(PyObject*, PyObject* args) {
    PyObject* x;
    int compile_time_errors;
    if (!PyArg_ParseTuple(args, "Op:_tensorlike_positive", &x, &compile_time_errors))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    return release_pyobject_or_raise(tensorlike_positive(x_val), compile_time_errors);
}

static Expected<Value> tensorlike_invert(BorrowedValue x) {
    if (x.is_loosely_typed_scalar()) {
        PyObject* x_obj = x.as_loosely_typed_scalar();
        if (PyBool_Check(x_obj))
            return make_error(Error::kTypeError, "`~` is not supported for boolean constants;"
                              " use ct.bitwise_not() instead");
        if (!PyLong_Check(x_obj))
            return make_error(Error::kTypeError,
                              "Operand of `~` must be an integer or boolean, not ", use_type(x));
        return loosely_typed_unary_result(PyNumber_Invert(x_obj));
    }
    if (!x.is_tensor())
        return make_error(Error::kTypeError,
                          "Operand of `~` must be an integer or boolean, not ", use_type(x));
    Expected<RefPtr<TensorBase>> res = tensor_invert(x.as_tensor());
    if (!res.is_ok()) return std::move(res.error());
    return Value::from_tensor(std::move(*res));
}

static PyObject* py_tensorlike_invert(PyObject*, PyObject* args) {
    PyObject* x;
    int compile_time_errors;
    if (!PyArg_ParseTuple(args, "Op:_tensorlike_invert", &x, &compile_time_errors))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    return release_pyobject_or_raise(tensorlike_invert(x_val), compile_time_errors);
}


// ----- Vector implementation -----

// Emits a getitem operation. The index is cast to int32 on the Python side.
static RefPtr<TensorBase> vector_getitem_symbolic(Vector* vector, TensorBase* index) {
    PyPtr vector_var = tensor_to_ir_var(vector);
    if (!vector_var) return {};
    PyPtr index_var = tensor_to_ir_var(index);
    if (!index_var) return {};

    static LazyImporter getitem_func;
    PyObject* func = getitem_func.get(
            "cuda.lang._ir.op_impl.vector_impl", "vector_getitem_raw");
    if (!func) return {};
    PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
            func, vector_var.get(), index_var.get(), nullptr));
    if (!result_var) return {};

    RefPtr<TensorBase> elem = new_0d_tensor(vector->dtype);
    if (!elem) return {};
    elem->data_.var = result_var.release();
    elem->tag_ = TensorBase::kVar;
    return elem;
}

// `index` must be in range
static RefPtr<TensorBase> vector_getitem_constant_index(Vector* vector, size_t index) {
    switch (vector->tag_) {
    case TensorBase::kBits:
    case TensorBase::kBitsArray:
        {
            RefPtr<TensorBase> elem = new_0d_tensor(vector->dtype);
            if (!elem) return {};
            elem->data_.bits = vector->tag_ == TensorBase::kBits
                    ? vector->data_.bits : vector->data_.bits_array[index];
            return elem;
        }
    case TensorBase::kVar:
        {
            RefPtr<TensorBase> index_tensor = tensor_new_scalar_or_0d_tile(
                    k_int32, static_cast<Bits>(index), false);
            if (!index_tensor) return {};
            return vector_getitem_symbolic(vector, index_tensor.get());
        }
    }
    CHECK_UNREACHABLE;
}

// Names a value in error messages, e.g. "Shape element #1", "Slice stop" or "Vector index"
struct Subject {
    const char* name;
    std::optional<size_t> index;  // printed as " #<index>" if present
};

static inline int append_to_string_builder(const Subject& subject, StringBuilder* sb) {
    sb->append(subject.name);
    if (subject.index.has_value())
        sb->append_many(" #", *subject.index);
    return 0;
}

// Returns the value of a Python int or a constant zero-dimensional integer tensor.
// If `overflow` is null, a value outside the int64_t range raises "<subject> is out of range".
// Otherwise, *overflow is set to 0, or to 1 / -1 if the value is above / below the range,
// in which case INT64_MAX / INT64_MIN is returned.
static Expected<int64_t> require_constant_int(BorrowedValue val, const Subject& subject,
                                              int* overflow = nullptr) {
    int value_overflow = 0;
    int64_t value = 0;
    if (val.is_loosely_typed_int()) {
        value = pylong_as_overflow_and<int64_t>(val.as_loosely_typed_scalar(), &value_overflow);
        if (PyErr_Occurred()) return ErrorRaised;
    } else if (val.is_tensor() && is_integer_dtype(val.tensor_dtype())
               && val.as_tensor()->shape().ndim() == 0) {
        TensorBase* tensor = val.as_tensor();
        if (tensor->is_symbolic())
            return make_error(Error::kTypeError, subject, " must be constant");
        // Compare without narrowing either side, so that this doesn't depend on
        // the widths of Bits and SignedBits relative to int64_t
        Bits bits = zero_dim_constant_bits_unchecked(tensor);
        if (is_unsigned_integer_dtype(tensor->dtype)) {
            if (bits > static_cast<uint64_t>(INT64_MAX))
                value_overflow = 1;
            else
                value = static_cast<int64_t>(bits);
        } else {
            SignedBits signed_bits = SignedBits(bits);
            if (signed_bits > INT64_MAX)
                value_overflow = 1;
            else if (signed_bits < INT64_MIN)
                value_overflow = -1;
            else
                value = static_cast<int64_t>(signed_bits);
        }
    } else {
        return make_error(Error::kTypeError, subject, " must be an integer, not ",
                          use_type(val));
    }

    if (overflow) *overflow = value_overflow;
    if (value_overflow == 0)
        return value;
    if (!overflow)
        return make_error(Error::kOverflowError, subject, " is out of range");
    return value_overflow > 0 ? INT64_MAX : INT64_MIN;
}

// Like PySlice_Unpack(), clamps out-of-range values
static Expected<Py_ssize_t> slice_index_value(const Value& index, const Subject& subject,
                                              Py_ssize_t default_value) {
    BorrowedValue val = index.borrow();
    if (val.is_none())
        return default_value;
    if (!val.is_loosely_typed_int() && !val.is_tensor())
        return make_error(Error::kTypeError, subject, " must be an integer or None, not ",
                          use_type(val));
    int overflow;
    Expected<int64_t> ret = require_constant_int(val, subject, &overflow);
    if (!ret.is_ok()) return std::move(ret.error());
    // Safe to ignore the overflow flag since the value is clamped to INT64_MIN/MAX.
    static_assert(PY_SSIZE_T_MAX == INT64_MAX && PY_SSIZE_T_MIN == INT64_MIN);
    return static_cast<Py_ssize_t>(*ret);
}

// Like cuda.lang's vector slicing, which doesn't allow an empty result
static Expected<RefPtr<TensorBase>> vector_slice(Vector* vector, const SliceValue& slice) {
    Expected<Py_ssize_t> step = slice_index_value(slice.step, Subject{"Slice step"}, 1);
    if (!step.is_ok()) return std::move(step.error());
    if (*step == 0)
        return make_error(Error::kValueError, "Slice step cannot be zero");
    // Like PySlice_Unpack(), so that PySlice_AdjustIndices() can safely negate the step
    if (*step < -PY_SSIZE_T_MAX)
        *step = -PY_SSIZE_T_MAX;
    // Same defaults as PySlice_Unpack()
    Expected<Py_ssize_t> start = slice_index_value(
            slice.start, Subject{"Slice start"}, *step > 0 ? 0 : PY_SSIZE_T_MAX);
    if (!start.is_ok()) return std::move(start.error());
    Expected<Py_ssize_t> stop = slice_index_value(
            slice.stop, Subject{"Slice stop"}, *step > 0 ? PY_SSIZE_T_MAX : PY_SSIZE_T_MIN);
    if (!stop.is_ok()) return std::move(stop.error());

    Py_ssize_t length = PySlice_AdjustIndices(vector->length, &*start, &*stop, *step);
    if (length == 0)
        return make_error(Error::kValueError, "Slice would result in an empty vector");

    size_t n = static_cast<size_t>(length);
    RefPtr<Vector> res = vector_zeros(vector->dtype, length);
    if (!res) return ErrorRaised;

    switch (vector->tag_) {
    case TensorBase::kBits:
        res->data_.bits = vector->data_.bits;
        break;
    case TensorBase::kBitsArray: {
        if (!res->allocate_bits_array(length)) return ErrorRaised;
        Bits* data = res->data_.bits_array;
        for (size_t i = 0; i < n; ++i)
            data[i] = vector->data_.bits_array[*start + static_cast<Py_ssize_t>(i) * *step];
        break;
    }
    case TensorBase::kVar: {
        PyPtr indices = steal(PyTuple_New(length));
        if (!indices) return ErrorRaised;
        for (size_t i = 0; i < n; ++i) {
            PyObject* index = PyLong_FromSsize_t(*start + static_cast<Py_ssize_t>(i) * *step);
            if (!index) return ErrorRaised;
            PyTuple_SET_ITEM(indices.get(), i, index);
        }

        static LazyImporter select_func;
        PyObject* func = select_func.get(
                "cuda.lang._ir.op_impl.vector_impl", "vector_select_elements");
        if (!func) return ErrorRaised;
        PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
                func, vector->var(), indices.get(), nullptr));
        if (!result_var) return ErrorRaised;

        res->data_.var = result_var.release();
        res->tag_ = TensorBase::kVar;
        break;
    }
    }
    return RefPtr<TensorBase>(std::move(res));
}

static Expected<Value> vector_getitem(BorrowedValue self, BorrowedValue key) {
    CHECK(self.is_tensor() && self.as_tensor()->kind == TensorBase::kVector);
    Vector* vector = static_cast<Vector*>(self.as_tensor());

    RefPtr<TensorBase> res;
    if (key.is_slice()) {
        RefPtr<SliceValue> slice = key.as_slice();
        Expected<RefPtr<TensorBase>> sliced = vector_slice(vector, *slice);
        if (!sliced.is_ok()) return std::move(sliced.error());
        res = std::move(*sliced);
    } else if (key.is_tensor() && key.as_tensor()->is_symbolic()
               && is_integer_dtype(key.tensor_dtype()) && key.as_tensor()->shape().ndim() == 0) {
        res = vector_getitem_symbolic(vector, key.as_tensor());
    } else if (key.is_loosely_typed_int() || key.is_tensor()) {
        int overflow;
        Expected<int64_t> index = require_constant_int(key, Subject{"Vector index"}, &overflow);
        if (!index.is_ok()) return std::move(index.error());
        // Negative constant indices are not wrapped around, consistent with run-time indexing
        if (overflow || *index < 0 || *index >= vector->length)
            return make_error(Error::kIndexError, "Vector index ", key,
                              " is out of range for a vector of length ", vector->length);
        res = vector_getitem_constant_index(vector, static_cast<size_t>(*index));
    } else {
        return make_error(Error::kTypeError, "Vector index must be an integer or a slice, not ",
                          use_type(key));
    }

    if (!res) return ErrorRaised;
    return Value::from_tensor(std::move(res));
}

static PyObject* vector_mp_subscript(PyObject* self, PyObject* key) {
    BorrowedValue key_val = borrowed_value_from_pyobject(key);
    BorrowedValue self_val = BorrowedValue::from_tensor(reinterpret_cast<TensorBase*>(self));
    Result<Value> res = to_result(vector_getitem(self_val, key_val));
    if (!res.is_ok()) return nullptr;
    return res->borrow().to_pyobject().release();
}

static Py_ssize_t vector_mp_length(PyObject* self) {
    return static_cast<Py_ssize_t>(reinterpret_cast<Vector*>(self)->length);
}

static int vector_mp_ass_subscript(PyObject*, PyObject*, PyObject* value) {
    if (value)
        raise(PyExc_TypeError,
              "Vectors are immutable. Consider calling vector.with_item() instead");
    else
        raise(PyExc_TypeError, "Vectors are immutable, so their items cannot be deleted");
    return -1;
}

static PyMappingMethods vector_mapping_methods = {
    .mp_length = vector_mp_length,
    .mp_subscript = vector_mp_subscript,
    .mp_ass_subscript = vector_mp_ass_subscript,
};

// Used for iteration, e.g. `for x in vector` or `tuple(vector)`
static PyObject* vector_sq_item(PyObject* self, Py_ssize_t index) {
    Vector* vector = reinterpret_cast<Vector*>(self);
    if (index < 0 || index >= vector->length) {
        raise(PyExc_IndexError, "Vector index ", index,
              " is out of range for a vector of length ", vector->length);
        return nullptr;
    }
    RefPtr<TensorBase> res = vector_getitem_constant_index(
            vector, static_cast<size_t>(index));
    return reinterpret_cast<PyObject*>(res.release());
}

// sq_length is intentionally not set: otherwise, PySequence_GetItem() would wrap around
// negative indices, unlike vector_getitem()
static PySequenceMethods vector_sequence_methods = {
    .sq_item = vector_sq_item,
};

// `index` must be in range
static Expected<Value> constant_vector_with_item(Vector* vector, size_t index, Bits item) {
    RefPtr<Vector> res = vector_zeros(vector->dtype, vector->length);
    if (!res) return ErrorRaised;
    if (!res->allocate_bits_array(vector->length)) return ErrorRaised;
    Bits* data = res->data_.bits_array;
    size_t length = static_cast<size_t>(vector->length);
    switch (vector->tag_) {
    case TensorBase::kBits:
        for (size_t i = 0; i < length; ++i)
            data[i] = vector->data_.bits;
        break;
    case TensorBase::kBitsArray:
        mem_copy(data, vector->data_.bits_array, length * sizeof(Bits));
        break;
    case TensorBase::kVar:
        CHECK_UNREACHABLE;
    }
    data[index] = item;
    return Value::from_tensor(std::move(res));
}

// Emits a VectorInsert. `index` must be an int32 scalar, and `item` must have the vector's dtype.
static Expected<Value> symbolic_vector_with_item(Vector* vector, TensorBase* index,
                                                 TensorBase* item) {
    PyPtr vector_var = tensor_to_ir_var(vector);
    if (!vector_var) return ErrorRaised;
    PyPtr index_var = tensor_to_ir_var(index);
    if (!index_var) return ErrorRaised;
    PyPtr item_var = tensor_to_ir_var(item);
    if (!item_var) return ErrorRaised;

    static LazyImporter with_item_func;
    PyObject* func = with_item_func.get(
            "cuda.lang._ir.op_impl.vector_impl", "vector_with_item_raw");
    if (!func) return ErrorRaised;
    PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
            func, vector_var.get(), index_var.get(), item_var.get(), nullptr));
    if (!result_var) return ErrorRaised;

    RefPtr<TensorBase> res = tensor_symbolic_like(vector, vector->dtype,
                                                  std::move(result_var));
    if (!res) return ErrorRaised;
    return Value::from_tensor(std::move(res));
}

// Returns a copy of `vector` with the element at `index` replaced by `value`. Like in
// vector_with_item_raw() in vector_impl.py, the index is implicitly cast to int32, and the value
// to the vector's dtype.
static Expected<Value> vector_with_item(Vector* vector, BorrowedValue index,
                                        BorrowedValue value) {
    // Booleans are accepted, as in implicit_cast()
    if (!(index.is_integer_tensor_like() || index.is_boolean_tensor_like())
            || !index.is_0d_tensor_like())
        return make_error(Error::kTypeError, "Vector index must be an integer, not ",
                          use_type(index));
    if (!value.is_0d_tensor_like())
        return make_error(Error::kTypeError, "Vector item must be a scalar, not ",
                          use_type(value));

    Expected<RefPtr<TensorBase>> index_tensor = implicit_cast(index, k_int32, false);
    if (!index_tensor.is_ok()) return std::move(index_tensor.error());
    Expected<RefPtr<TensorBase>> item = implicit_cast(value, vector->dtype, false);
    if (!item.is_ok()) return std::move(item.error());

    if (!(*index_tensor)->is_symbolic()) {
        SignedBits i = SignedBits(zero_dim_constant_bits_unchecked(index_tensor->get()));
        // Negative indices are not wrapped around, consistent with vector_getitem()
        if (i < 0 || i >= vector->length)
            return make_error(Error::kIndexError, "Vector index ", i,
                              " is out of range for a vector of length ", vector->length);
        if (!vector->is_symbolic() && !(*item)->is_symbolic())
            return constant_vector_with_item(vector, static_cast<size_t>(i),
                                             zero_dim_constant_bits_unchecked(item->get()));
    }
    return symbolic_vector_with_item(vector, index_tensor->get(), item->get());
}

static PyObject* vector_method_with_item(PyObject* self, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"index", "value", nullptr};
    PyObject *index, *value;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:with_item", const_cast<char**>(keywords),
                                     &index, &value))
        return nullptr;

    BorrowedValue index_val = borrowed_value_from_pyobject(index);
    BorrowedValue value_val = borrowed_value_from_pyobject(value);
    return release_pyobject_or_raise(
            vector_with_item(reinterpret_cast<Vector*>(self), index_val, value_val),
            false);
}

// Raises NotImplementedError for operations that can't be evaluated outside of device code,
// e.g. on the host or inside static_eval(). `what` names the operation, e.g. "Tile.extract()";
// the message depends on the current DispatchMode.
static ErrorRaised_t unsupported_call_error(const char* what) {
    static LazyImporter get_current_func;
    PyObject* func = get_current_func.get("cuda.tile._dispatch_mode", "DispatchMode",
                                          "get_current");
    if (!func) return ErrorRaised;

    PyPtr mode = steal(PyObject_CallNoArgs(func));
    if (!mode) return ErrorRaised;
    PyPtr message = steal(PyObject_CallMethod(mode.get(), "unsupported_call_message", "s", what));
    if (!message) return ErrorRaised;
    if (!PyUnicode_Check(message.get()))
        return raise(PyExc_AssertionError, "unsupported_call_message() must return a str, not ",
                     Py_TYPE(message.get())->tp_name);
    PyErr_SetObject(PyExc_NotImplementedError, message.get());
    return ErrorRaised;
}

static PyObject* method_unsupported(const char* what) {
    unsupported_call_error(what);
    return nullptr;
}

static PyObject* vector_method_astype(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"dtype", nullptr};
    PyObject* dtype;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:astype", const_cast<char**>(keywords),
                                     &dtype))
        return nullptr;
    return method_unsupported("Vector.astype()");
}

static PyObject* vector_method_bitcast(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"dtype", nullptr};
    PyObject* dtype;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:bitcast", const_cast<char**>(keywords),
                                     &dtype))
        return nullptr;
    return method_unsupported("Vector.bitcast()");
}

static PyObject* vector_method_reinterpret_as_scalar(PyObject*, PyObject* args,
                                                     PyObject* kwargs) {
    static const char* keywords[] = {"dtype", nullptr};
    PyObject* dtype;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:reinterpret_as_scalar",
                                     const_cast<char**>(keywords), &dtype))
        return nullptr;
    return method_unsupported("Vector.reinterpret_as_scalar()");
}

static PyObject* vector_method_reinterpret_as_vector(PyObject*, PyObject* args,
                                                     PyObject* kwargs) {
    static const char* keywords[] = {"dtype", "length", nullptr};
    PyObject *dtype, *length;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:reinterpret_as_vector",
                                     const_cast<char**>(keywords), &dtype, &length))
        return nullptr;
    return method_unsupported("Vector.reinterpret_as_vector()");
}

static PyObject* vector_method_reduce(PyObject*, PyObject* args, PyObject* kwargs) {
    // `op` is positional-only, the flags are keyword-only
    static const char* keywords[] = {"", "propagate_nan", "reassociate", nullptr};
    PyObject* op;
    int propagate_nan = 0, reassociate = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$pp:reduce", const_cast<char**>(keywords),
                                     &op, &propagate_nan, &reassociate))
        return nullptr;
    return method_unsupported("Vector.reduce()");
}

static PyMethodDef vector_methods[] = {
    {"astype", reinterpret_cast<PyCFunction>(vector_method_astype),
     METH_VARARGS | METH_KEYWORDS, kDoc_Vector_astype},
    {"bitcast", reinterpret_cast<PyCFunction>(vector_method_bitcast),
     METH_VARARGS | METH_KEYWORDS, kDoc_Vector_bitcast},
    {"reduce", reinterpret_cast<PyCFunction>(vector_method_reduce),
     METH_VARARGS | METH_KEYWORDS, kDoc_Vector_reduce},
    {"reinterpret_as_scalar", reinterpret_cast<PyCFunction>(vector_method_reinterpret_as_scalar),
     METH_VARARGS | METH_KEYWORDS, kDoc_Vector_reinterpret_as_scalar},
    {"reinterpret_as_vector", reinterpret_cast<PyCFunction>(vector_method_reinterpret_as_vector),
     METH_VARARGS | METH_KEYWORDS, kDoc_Vector_reinterpret_as_vector},
    {"with_item", reinterpret_cast<PyCFunction>(vector_method_with_item),
     METH_VARARGS | METH_KEYWORDS, kDoc_Vector_with_item},
    {}
};

static PyObject* py_vector_getitem(PyObject*, PyObject* args) {
    PyObject *x, *key;
    if (!PyArg_ParseTuple(args, "O!O:_vector_getitem", &Vector::pytype, &x, &key))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    BorrowedValue key_val = borrowed_value_from_pyobject(key);
    return release_pyobject_or_raise(vector_getitem(x_val, key_val), true);
}

static PyObject* py_vector_with_item(PyObject*, PyObject* args) {
    PyObject *x, *index, *value;
    if (!PyArg_ParseTuple(args, "O!OO:_vector_with_item", &Vector::pytype, &x, &index, &value))
        return nullptr;

    BorrowedValue index_val = borrowed_value_from_pyobject(index);
    BorrowedValue value_val = borrowed_value_from_pyobject(value);
    return release_pyobject_or_raise(
            vector_with_item(reinterpret_cast<Vector*>(x), index_val, value_val), true);
}

static PyObject* vector_get_element_count(PyObject* self, void*) {
    Vector* vector = reinterpret_cast<Vector*>(self);
    return PyLong_FromLongLong(vector->length);
}

static PyObject* tensor_get_dtype(PyObject* self, void*) {
    TensorBase* tensor = reinterpret_cast<TensorBase*>(self);
    return Py_XNewRef(dtype_to_pyobject(tensor->dtype));
}

static PyGetSetDef vector_getset[] = {
    {"dtype", tensor_get_dtype, nullptr, kDoc_Vector_dtype, nullptr},
    {"element_count", vector_get_element_count, nullptr, kDoc_Vector_element_count, nullptr},
    {}
};


// ----- Tile implementation -----

// Shapes for matrix multiplication, like numpy.matmul(): a 1-D `x` is treated as a row vector,
// a 1-D `y` as a column vector, and the leading (batch) dimensions are broadcast
struct MatmulShapes {
    UnpackedShape x;  // (*batch, M, K)
    UnpackedShape y;  // (*batch, K, N)
    UnpackedShape product;  // (*batch, M, N)
    UnpackedShape result;  // `product` without the dimensions added to 1-D operands
};

static Expected<MatmulShapes> matmul_shapes(const UnpackedShape& x, const UnpackedShape& y) {
    CHECK(x.ndim() > 0 && y.ndim() > 0);
    auto incompatible = [&] () {
        return make_error(Error::kTypeError, "Shapes ", x, " and ", y,
                          " are incompatible for matrix multiplication");
    };

    int64_t m = x.ndim() > 1 ? x[x.ndim() - 2] : 1;
    int64_t k = x[x.ndim() - 1];
    int64_t y_k = y.ndim() > 1 ? y[y.ndim() - 2] : y[0];
    int64_t n = y.ndim() > 1 ? y[y.ndim() - 1] : 1;
    if (k != y_k) return incompatible();

    size_t x_batch_ndim = x.ndim() > 1 ? x.ndim() - 2 : 0;
    size_t y_batch_ndim = y.ndim() > 1 ? y.ndim() - 2 : 0;
    size_t batch_ndim = x_batch_ndim > y_batch_ndim ? x_batch_ndim : y_batch_ndim;
    TemporaryArray<int64_t, PackedShape::kMaxInlineDims> dims_buf;
    int64_t* dims = dims_buf.allocate(batch_ndim + 2);

    // Batch dimensions are aligned to the right
    for (size_t i = 0; i < batch_ndim; ++i) {
        size_t x_offset = batch_ndim - x_batch_ndim;
        size_t y_offset = batch_ndim - y_batch_ndim;
        int64_t x_dim = i >= x_offset ? x[i - x_offset] : 1;
        int64_t y_dim = i >= y_offset ? y[i - y_offset] : 1;
        if (x_dim != y_dim && x_dim != 1 && y_dim != 1) return incompatible();
        dims[i] = x_dim == 1 ? y_dim : x_dim;
    }

    // Each shape reuses the batch dimensions, followed by its own trailing dimensions
    dims[batch_ndim] = m;
    dims[batch_ndim + 1] = k;
    Expected<UnpackedShape> x_shape = UnpackedShape::create(batch_ndim + 2, dims);
    if (!x_shape.is_ok()) return std::move(x_shape.error());

    dims[batch_ndim] = k;
    dims[batch_ndim + 1] = n;
    Expected<UnpackedShape> y_shape = UnpackedShape::create(batch_ndim + 2, dims);
    if (!y_shape.is_ok()) return std::move(y_shape.error());

    dims[batch_ndim] = m;
    dims[batch_ndim + 1] = n;
    Expected<UnpackedShape> product_shape = UnpackedShape::create(batch_ndim + 2, dims);
    if (!product_shape.is_ok()) return std::move(product_shape.error());

    size_t result_ndim = batch_ndim;
    if (x.ndim() > 1) dims[result_ndim++] = m;
    if (y.ndim() > 1) dims[result_ndim++] = n;
    Expected<UnpackedShape> result_shape = UnpackedShape::create(result_ndim, dims);
    if (!result_shape.is_ok()) return std::move(result_shape.error());

    return MatmulShapes{*x_shape, *y_shape, *product_shape, *result_shape};
}

static RefPtr<TensorBase> tile_reshape_unchecked(Tile* tile, const UnpackedShape& new_shape) {
    RefPtr<TensorBase> ret = tensor_zeros_with_shape(tile->dtype, new_shape, true);
    if (!ret) return {};
    switch (tile->tag_) {
    case TensorBase::kBits:
        ret->data_.bits = tile->data_.bits;
        break;
    case TensorBase::kBitsArray:
        {
            CHECK(ret->numel() == tile->numel());
            size_t numel = static_cast<size_t>(ret->numel());
            if (!ret->allocate_bits_array(ret->numel())) return {};
            Bits* data = ret->data_.bits_array;
            mem_copy(data, tile->data_.bits_array, numel * sizeof(Bits));
            break;
        }
    case TensorBase::kVar:
        {
            static LazyImporter reshape_func;
            PyObject* func = reshape_func.get(
                    "cuda.tile._ir.arithmetic_ops", "reshape_raw");
            if (!func) return {};
            PyPtr py_shape = shape_to_pytuple(new_shape);
            if (!py_shape) return {};
            PyPtr result_var = steal(PyObject_CallFunctionObjArgs(
                    func, tile->var(), py_shape.get(), nullptr));
            if (!result_var) return {};
            ret->data_.var = result_var.release();
            ret->tag_ = TensorBase::kVar;
            break;
        }
    }
    return ret;
}


static Error require_symbolic_evaluation(const char* message) {
    static LazyImporter have_current_func;
    PyObject* func = have_current_func.get("cuda.tile._ir.ir", "Builder", "have_current");
    if (!func) return ErrorRaised;

    PyPtr have_current = steal(PyObject_CallNoArgs(func));
    if (!have_current) return ErrorRaised;
    if (have_current.get() != Py_True)
        return make_error(Error::kNotImplementedError, message);
    return OK;
}


// The operation is emitted even for constant tiles
static Expected<Value> tile_matmul(BorrowedValue x, BorrowedValue y) {
    RefPtr<TensorBase> x_tensor, y_tensor;
    if (!prepare_tensorlike_operands(x, y, &x_tensor, &y_tensor))
        return Value::not_implemented();
    Error err = require_symbolic_evaluation(
            "Matrix multiplication is only supported inside a kernel");
    if (!err.is_ok()) return std::move(err);
    if (!x_tensor || x_tensor->kind != TensorBase::kTile || x_tensor->shape().ndim() == 0)
        return make_error(Error::kTypeError, "Left-hand side of matrix multiplication must"
                          " be a tile with at least one dimension, not ", use_type(x));
    if (!y_tensor || y_tensor->kind != TensorBase::kTile || y_tensor->shape().ndim() == 0)
        return make_error(Error::kTypeError, "Right-hand side of matrix multiplication must"
                          " be a tile with at least one dimension, not ", use_type(y));
    if (!is_numeric_dtype(x_tensor->dtype))
        return make_error(Error::kTypeError, "Left-hand side of matrix multiplication must"
                          " have a numeric dtype, not ", x_tensor->dtype);
    if (!is_numeric_dtype(y_tensor->dtype))
        return make_error(Error::kTypeError, "Right-hand side of matrix multiplication must"
                          " have a numeric dtype, not ", y_tensor->dtype);

    UnpackedShape x_shape = x_tensor->shape().unpack();
    UnpackedShape y_shape = y_tensor->shape().unpack();
    Expected<MatmulShapes> shapes = matmul_shapes(x_shape, y_shape);
    if (!shapes.is_ok()) return std::move(shapes.error());

    Error promoted = promote_tensors(&x_tensor, &y_tensor, false);
    if (!promoted.is_ok()) return std::move(promoted);

    // A 1-D `y` becomes a column, which can't be done by broadcasting
    if (y_shape.ndim() == 1) {
        int64_t column_dims[2] = {y_shape[0], 1};
        Expected<UnpackedShape> column_shape = UnpackedShape::create(2, column_dims);
        if (!column_shape.is_ok()) return std::move(column_shape.error());
        y_tensor = tile_reshape_unchecked(static_cast<Tile*>(y_tensor.get()), *column_shape);
        if (!y_tensor) return ErrorRaised;
    }
    if (x_tensor->shape().unpack() != shapes->x) {
        Error err = broadcast_to_unchecked(&x_tensor, shapes->x);
        if (!err.is_ok()) return std::move(err);
    }
    if (y_tensor->shape().unpack() != shapes->y) {
        Error err = broadcast_to_unchecked(&y_tensor, shapes->y);
        if (!err.is_ok()) return std::move(err);
    }

    PyPtr x_var = tensor_to_ir_var(x_tensor.get());
    if (!x_var) return ErrorRaised;
    PyPtr y_var = tensor_to_ir_var(y_tensor.get());
    if (!y_var) return ErrorRaised;

    static LazyImporter matmul_func;
    PyObject* func = matmul_func.get("cuda.tile._ir.ops", "matmul_raw");
    if (!func) return ErrorRaised;
    PyPtr product_var = steal(PyObject_CallFunctionObjArgs(
            func, x_var.get(), y_var.get(), nullptr));
    if (!product_var) return ErrorRaised;

    RefPtr<TensorBase> product = tensor_zeros_with_shape(
            x_tensor->dtype, shapes->product, true);
    if (!product) return ErrorRaised;
    product->data_.var = product_var.release();
    product->tag_ = TensorBase::kVar;
    if (shapes->result == shapes->product)
        return Value::from_tensor(std::move(product));

    RefPtr<TensorBase> res = tile_reshape_unchecked(
            static_cast<Tile*>(product.get()), shapes->result);
    if (!res) return ErrorRaised;
    return Value::from_tensor(std::move(res));
}

static PyObject* tile_nb_matrix_multiply(PyObject* x, PyObject* y) {
    BorrowedValue xval = borrowed_value_from_pyobject(x);
    BorrowedValue yval = borrowed_value_from_pyobject(y);
    Result<Value> res = to_result(tile_matmul(xval, yval));
    if (!res.is_ok()) return nullptr;
    return res->borrow().to_pyobject().release();
}

static PyObject* py_tile_matmul(PyObject*, PyObject* args) {
    PyObject *x, *y;
    if (!PyArg_ParseTuple(args, "OO:_tile_matmul", &x, &y))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    BorrowedValue y_val = borrowed_value_from_pyobject(y);
    Expected<Value> res = tile_matmul(x_val, y_val);

    if (res.is_ok() && res->is_not_implemented()) {
        res = make_error(Error::kTypeError, "Unsupported operand types for @: '",
                use_type(x_val), "' and '", use_type(y_val), "'");
    }
    return release_pyobject_or_raise(std::move(res), true);
}


static bool is_full_slice(BorrowedValue val) {
    if (!val.is_slice()) return false;
    RefPtr<SliceValue> slice = val.as_slice();
    return slice->start.borrow().is_none() && slice->stop.borrow().is_none()
        && slice->step.borrow().is_none();
}

static Expected<Value> tile_getitem(BorrowedValue self, BorrowedValue key) {
    CHECK(self.is_tensor() && self.as_tensor()->kind == TensorBase::kTile);
    Tile* tile = static_cast<Tile*>(self.as_tensor());
    UnpackedShape shape = tile->shape_.unpack();

    // Normalize the key as a sequence of indices
    RefPtr<TupleValue> tuple;
    Value single;
    const Value* index;
    size_t index_size;
    if (key.is_tuple()) {
        tuple = key.as_tuple();
        index = tuple->items();
        index_size = tuple->size();
    } else {
        // Same as tile[key,]
        single = key.to_owned();
        index = &single;
        index_size = 1;
    }

    // Parse the indices
    size_t num_slices = 0, num_nones = 0, num_ellipses = 0;
    for (size_t i = 0; i < index_size; ++i) {
        BorrowedValue item = index[i].borrow();
        if (item.is_none()) {
            ++num_nones;
            continue;
        }
        if (item.is_ellipsis()) {
            ++num_ellipses;
            continue;
        }
        if (is_full_slice(item)) {
            ++num_slices;
            continue;
        }
        return make_error(Error::kTypeError,
                          "Tile index must be None, an ellipsis or a full slice (`:`), not ",
                          use_type(item), ". Hint: to index a tile, use `extract()` or `item()`.");
    }

    if (num_slices > shape.ndim())
        return make_error(Error::kTypeError, "Tile index has ", num_slices,
                          " full slices, but the tile has only ", shape.ndim(), " dimensions");
    if (num_ellipses > 1)
        return make_error(Error::kTypeError, "Tile index has more than one ellipsis");

    // Compute the shape
    size_t new_ndim = shape.ndim() + num_nones;
    TemporaryArray<int64_t, PackedShape::kMaxInlineDims> new_dims_buf;
    int64_t* new_dims = new_dims_buf.allocate(new_ndim);
    size_t in = 0, out = 0;
    for (size_t i = 0; i < index_size; ++i) {
        BorrowedValue item = index[i].borrow();
        if (item.is_none()) {
            new_dims[out++] = 1;
        } else if (item.is_ellipsis()) {
            for (size_t j = 0; j < shape.ndim() - num_slices; ++j)
                new_dims[out++] = shape[in++];
        } else {
            new_dims[out++] = shape[in++];
        }
    }
    while (in < shape.ndim())
        new_dims[out++] = shape[in++];
    CHECK(out == new_ndim);

    // Reshape to the new shape
    Expected<UnpackedShape> new_shape = UnpackedShape::create(new_ndim, new_dims);
    if (!new_shape.is_ok()) return std::move(new_shape.error());
    RefPtr<TensorBase> res = tile_reshape_unchecked(tile, *new_shape);
    if (!res) return ErrorRaised;
    return Value::from_tensor(std::move(res));
}

static Expected<Value> tile_reshape(Tile* tile, BorrowedValue shape_arg) {
    if (!shape_arg.is_tuple())
        return make_error(Error::kTypeError, "Shape must be a tuple, not ",
                          use_type(shape_arg));
    RefPtr<TupleValue> tuple = shape_arg.as_tuple();

    UnpackedShape shape = tile->shape_.unpack();
    auto cannot_reshape = [&] () {
        return make_error(Error::kTypeError, "Cannot reshape ", shape, " to ", *tuple);
    };

    size_t new_ndim = tuple->size();
    TemporaryArray<int64_t, PackedShape::kMaxInlineDims> new_dims_buf;
    int64_t* new_dims = new_dims_buf.allocate(new_ndim);
    std::optional<size_t> inferred_dim;
    int64_t known_numel = 1;  // product of all dimensions except the inferred one
    for (size_t i = 0; i < new_ndim; ++i) {
        Expected<int64_t> r = require_constant_int((*tuple)[i].borrow(),
                                                   Subject{"Shape element", i});
        if (!r.is_ok()) return std::move(r.error());
        int64_t value = *r;

        if (value == -1) {
            if (inferred_dim.has_value())
                return make_error(Error::kTypeError, "Shape ", *tuple, " has more than one -1");
            inferred_dim = i;
        } else if (value <= 0) {
            return make_error(Error::kTypeError, "Shape element #", i,
                              " must be positive or -1, not ", value);
        } else {
            if (int64_mul_overflow(known_numel, value, &known_numel))
                return make_error(Error::kTypeError, "Shape ", *tuple,
                                  " has too many elements");
            new_dims[i] = value;
        }
    }

    int64_t numel = tile->numel();
    if (inferred_dim.has_value()) {
        if (numel % known_numel != 0)
            return cannot_reshape();
        new_dims[*inferred_dim] = numel / known_numel;
    } else if (numel != known_numel) {
        return cannot_reshape();
    }

    Expected<UnpackedShape> new_shape = UnpackedShape::create(new_ndim, new_dims);
    if (!new_shape.is_ok()) return std::move(new_shape.error());
    if (*new_shape == shape)
        return Value::from_tensor(newref(static_cast<TensorBase*>(tile)));

    RefPtr<TensorBase> res = tile_reshape_unchecked(tile, *new_shape);
    if (!res) return ErrorRaised;
    return Value::from_tensor(std::move(res));
}

// Equivalent to tile.reshape(())
static Expected<RefPtr<Tile>> tile_item(Tile* tile) {
    if (tile->shape_.ndim() == 0)
        return newref(tile);
    if (tile->numel() != 1)
        return make_error(Error::kTypeError, "Cannot reshape ", tile->shape_, " to ()");

    RefPtr<TensorBase> res = tile_reshape_unchecked(tile, UnpackedShape());
    if (!res) return ErrorRaised;
    return steal(static_cast<Tile*>(res.release()));
}

static PyObject* tile_method_item(PyObject* self, PyObject*) {
    Result<RefPtr<Tile>> res = to_result(tile_item(reinterpret_cast<Tile*>(self)));
    if (!res.is_ok()) return nullptr;
    return reinterpret_cast<PyObject*>(res->release());
}

static PyObject* tile_method_reshape(PyObject* self, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"shape", nullptr};
    PyObject* shape;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:reshape", const_cast<char**>(keywords),
                                     &shape))
        return nullptr;

    BorrowedValue shape_val = borrowed_value_from_pyobject(shape);
    Result<Value> res = to_result(tile_reshape(reinterpret_cast<Tile*>(self), shape_val));
    if (!res.is_ok()) return nullptr;
    return res->borrow().to_pyobject().release();
}

static PyObject* tile_method_extract(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"index", "shape", nullptr};
    PyObject *index, *shape;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:extract", const_cast<char**>(keywords),
                                     &index, &shape))
        return nullptr;
    return method_unsupported("Tile.extract()");
}

static PyObject* tile_method_insert(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"index", "value", nullptr};
    PyObject *index, *value;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:insert", const_cast<char**>(keywords),
                                     &index, &value))
        return nullptr;
    return method_unsupported("Tile.insert()");
}

static PyObject* tile_method_permute(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"axes", nullptr};
    PyObject* axes;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:permute", const_cast<char**>(keywords),
                                     &axes))
        return nullptr;
    return method_unsupported("Tile.permute()");
}

static PyObject* tile_method_transpose(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"axis0", "axis1", nullptr};
    PyObject* axis0 = Py_None;
    PyObject* axis1 = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO:transpose", const_cast<char**>(keywords),
                                     &axis0, &axis1))
        return nullptr;
    return method_unsupported("Tile.transpose()");
}

static PyObject* tile_method_astype(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"dtype", "rounding_mode", nullptr};
    PyObject* dtype;
    PyObject* rounding_mode = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O:astype", const_cast<char**>(keywords),
                                     &dtype, &rounding_mode))
        return nullptr;
    return method_unsupported("Tile.astype()");
}

static PyMethodDef tile_methods[] = {
    {"astype", reinterpret_cast<PyCFunction>(tile_method_astype),
     METH_VARARGS | METH_KEYWORDS, kDoc_Tile_astype},
    {"extract", reinterpret_cast<PyCFunction>(tile_method_extract),
     METH_VARARGS | METH_KEYWORDS, kDoc_Tile_extract},
    {"insert", reinterpret_cast<PyCFunction>(tile_method_insert),
     METH_VARARGS | METH_KEYWORDS, kDoc_Tile_insert},
    {"item", tile_method_item, METH_NOARGS, kDoc_Tile_item},
    {"permute", reinterpret_cast<PyCFunction>(tile_method_permute),
     METH_VARARGS | METH_KEYWORDS, kDoc_Tile_permute},
    {"reshape", reinterpret_cast<PyCFunction>(tile_method_reshape),
     METH_VARARGS | METH_KEYWORDS, kDoc_Tile_reshape},
    {"transpose", reinterpret_cast<PyCFunction>(tile_method_transpose),
     METH_VARARGS | METH_KEYWORDS, kDoc_Tile_transpose},
    {}
};

static PyObject* tile_mp_subscript(PyObject* self, PyObject* key) {
    BorrowedValue key_val = borrowed_value_from_pyobject(key);
    BorrowedValue self_val = BorrowedValue::from_tensor(reinterpret_cast<TensorBase*>(self));
    Result<Value> res = to_result(tile_getitem(self_val, key_val));
    if (!res.is_ok()) return nullptr;
    return res->borrow().to_pyobject().release();
}

static int tile_mp_ass_subscript(PyObject*, PyObject*, PyObject* value) {
    if (value)
        raise(PyExc_TypeError, "Tiles are immutable. Consider calling tile.insert() instead");
    else
        raise(PyExc_TypeError, "Tiles are immutable, so their items cannot be deleted");
    return -1;
}

static PyMappingMethods tile_mapping_methods = {
    .mp_subscript = tile_mp_subscript,
    .mp_ass_subscript = tile_mp_ass_subscript,
};

// A Python number becomes a 0-d tile of its default dtype
static Expected<RefPtr<Tile>> tilelike_to_tile(BorrowedValue x, const char* operand) {
    if (x.is_loosely_typed_scalar()) {
        Expected<RefPtr<TensorBase>> r = loosely_typed_scalar_to_default_dtype(x, true);
        if (!r.is_ok()) return std::move(r.error());
        return steal(static_cast<Tile*>(r->release()));
    }
    if (x.is_tensor() && x.as_tensor()->kind == TensorBase::kTile)
        return newref(static_cast<Tile*>(x.as_tensor()));
    return make_error(Error::kTypeError, operand, " must be a tile, not ", use_type(x));
}

static Expected<Value> tilelike_reshape(BorrowedValue x, BorrowedValue shape) {
    Expected<RefPtr<Tile>> tile = tilelike_to_tile(x, "Reshape operand");
    if (!tile.is_ok()) return std::move(tile.error());
    return tile_reshape(tile->get(), shape);
}

static Expected<Value> tilelike_getitem(BorrowedValue x, BorrowedValue key) {
    Expected<RefPtr<Tile>> tile = tilelike_to_tile(x, "Indexed object");
    if (!tile.is_ok()) return std::move(tile.error());
    return tile_getitem(BorrowedValue::from_tensor(tile->get()), key);
}

static Expected<RefPtr<Tile>> tilelike_item(BorrowedValue x) {
    Expected<RefPtr<Tile>> tile = tilelike_to_tile(x, "Operand of item()");
    if (!tile.is_ok()) return std::move(tile.error());
    return tile_item(tile->get());
}

static PyObject* py_tile_reshape(PyObject*, PyObject* args) {
    PyObject *x, *shape;
    if (!PyArg_ParseTuple(args, "OO:_tile_reshape", &x, &shape))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    BorrowedValue shape_val = borrowed_value_from_pyobject(shape);
    return release_pyobject_or_raise(tilelike_reshape(x_val, shape_val), true);
}

static PyObject* py_tile_getitem(PyObject*, PyObject* args) {
    PyObject *x, *key;
    if (!PyArg_ParseTuple(args, "OO:_tile_getitem", &x, &key))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    BorrowedValue key_val = borrowed_value_from_pyobject(key);
    return release_pyobject_or_raise(tilelike_getitem(x_val, key_val), true);
}

static PyObject* py_tile_item(PyObject*, PyObject* args) {
    PyObject* x;
    if (!PyArg_ParseTuple(args, "O:_tile_item", &x))
        return nullptr;

    BorrowedValue x_val = borrowed_value_from_pyobject(x);
    return release_pyobject_or_raise(tilelike_item(x_val), true);
}

static PyObject* tile_get_shape(PyObject* self, void*) {
    Tile* tile = reinterpret_cast<Tile*>(self);
    return shape_to_pytuple(tile->shape_.unpack()).release();
}

static PyObject* tile_get_ndim(PyObject* self, void*) {
    Tile* tile = reinterpret_cast<Tile*>(self);
    return PyLong_FromSize_t(tile->shape_.ndim());
}

static PyGetSetDef tile_getset[] = {
    {"dtype", tensor_get_dtype, nullptr, kDoc_Tile_dtype, nullptr},
    {"shape", tile_get_shape, nullptr, kDoc_Tile_shape, nullptr},
    {"ndim", tile_get_ndim, nullptr, kDoc_Tile_ndim, nullptr},
    {}
};


// ----- Pointer implementation -----

static PyObject* pointer_get_opaque(PyObject* self, void*) {
    TensorBase* pointer = reinterpret_cast<TensorBase*>(self);
    return PyBool_FromLong(!pointer_dtype_pointee(pointer->dtype));
}

static PyObject* pointer_get_pointee_dtype(PyObject* self, void*) {
    TensorBase* pointer = reinterpret_cast<TensorBase*>(self);
    DType pointee = pointer_dtype_pointee(pointer->dtype);
    if (!pointee) {
        raise(PyExc_TypeError, "Opaque pointer has no pointee_dtype");
        return nullptr;
    }
    return Py_XNewRef(dtype_to_pyobject(pointee));
}

static PyObject* pointer_get_memory_space(PyObject* self, void*) {
    TensorBase* pointer = reinterpret_cast<TensorBase*>(self);
    MemorySpace space = pointer_dtype_memory_space(pointer->dtype);
    return Py_XNewRef(enum_to_pyobject(space));
}

static PyObject* pointer_method_load(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"count", "alignment", nullptr};
    PyObject* count = nullptr;
    PyObject* alignment = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|$OO:load", const_cast<char**>(keywords),
                                     &count, &alignment))
        return nullptr;
    return method_unsupported("Pointer.load()");
}

static PyObject* pointer_method_store(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"value", "alignment", nullptr};
    PyObject* value;
    PyObject* alignment = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O:store", const_cast<char**>(keywords),
                                     &value, &alignment))
        return nullptr;
    return method_unsupported("Pointer.store()");
}

static PyObject* pointer_method_atomic_load(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"count", "memory_order", "memory_scope", "mmio", "alignment",
                                     nullptr};
    PyObject* count = nullptr;
    PyObject *memory_order = nullptr, *memory_scope = nullptr;
    PyObject *mmio = Py_False, *alignment = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|$OOOOO:atomic_load",
                                     const_cast<char**>(keywords),
                                     &count, &memory_order, &memory_scope, &mmio, &alignment))
        return nullptr;
    return method_unsupported("Pointer.atomic_load()");
}

static PyObject* pointer_method_atomic_store(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"value", "memory_order", "memory_scope", "mmio",
                                     "alignment", nullptr};
    PyObject* value;
    PyObject *memory_order = nullptr, *memory_scope = nullptr;
    PyObject *mmio = Py_False, *alignment = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$OOOO:atomic_store",
                                     const_cast<char**>(keywords),
                                     &value, &memory_order, &memory_scope, &mmio, &alignment))
        return nullptr;
    return method_unsupported("Pointer.atomic_store()");
}

static PyMethodDef pointer_methods[] = {
    {"atomic_load", reinterpret_cast<PyCFunction>(pointer_method_atomic_load),
     METH_VARARGS | METH_KEYWORDS, kDoc_Pointer_atomic_load},
    {"atomic_store", reinterpret_cast<PyCFunction>(pointer_method_atomic_store),
     METH_VARARGS | METH_KEYWORDS, kDoc_Pointer_atomic_store},
    {"load", reinterpret_cast<PyCFunction>(pointer_method_load),
     METH_VARARGS | METH_KEYWORDS, kDoc_Pointer_load},
    {"store", reinterpret_cast<PyCFunction>(pointer_method_store),
     METH_VARARGS | METH_KEYWORDS, kDoc_Pointer_store},
    {}
};

// `pointer[index]` is equivalent to `(pointer + index).load()`
static PyObject* pointer_mp_subscript(PyObject*, PyObject*) {
    return method_unsupported("Pointer indexing");
}

// `pointer[index] = value` is equivalent to `(pointer + index).store(value)`.
// Also called with value == nullptr for `del pointer[index]`.
static int pointer_mp_ass_subscript(PyObject*, PyObject*, PyObject* value) {
    if (value)
        method_unsupported("Pointer item assignment");
    else
        raise(PyExc_TypeError, "Pointer items cannot be deleted");
    return -1;
}

static PyMappingMethods pointer_mapping_methods = {
    .mp_subscript = pointer_mp_subscript,
    .mp_ass_subscript = pointer_mp_ass_subscript,
};

static PyGetSetDef pointer_getset[] = {
    {"pointer_dtype", tensor_get_dtype, nullptr, kDoc_Pointer_pointer_dtype, nullptr},
    {"opaque", pointer_get_opaque, nullptr, kDoc_Pointer_opaque, nullptr},
    {"pointee_dtype", pointer_get_pointee_dtype, nullptr, kDoc_Pointer_pointee_dtype, nullptr},
    {"memory_space", pointer_get_memory_space, nullptr, kDoc_Pointer_memory_space, nullptr},
    {}
};



// ----- Scalar implementation -----

static PyGetSetDef scalar_getset[] = {
    {"dtype", tensor_get_dtype, nullptr, kDoc_Scalar_dtype, nullptr},
    {}
};


// ----- OpaqueValue implementation -----

static PyGetSetDef opaque_value_getset[] = {
    {"dtype", tensor_get_dtype, nullptr, kDoc_OpaqueValue_dtype, nullptr},
    {}
};


// ----- Module initialization -----

// Number methods shared by all tensor types
#define TENSOR_NUMBER_METHODS \
    .nb_add = tensor_arith_binop<BinaryArithmeticOp::Add>, \
    .nb_subtract = tensor_arith_binop<BinaryArithmeticOp::Sub>, \
    .nb_multiply = tensor_arith_binop<BinaryArithmeticOp::Mul>, \
    .nb_remainder = tensor_arith_binop<BinaryArithmeticOp::Mod>, \
    .nb_divmod = tensor_nb_divmod, \
    .nb_power = tensor_nb_power, \
    .nb_negative = tensor_nb_negative, \
    .nb_positive = tensor_nb_positive, \
    .nb_bool = tensor_bool, \
    .nb_invert = tensor_nb_invert, \
    .nb_lshift = tensor_bitwise_shift_binop<BitwiseShiftOp::LShift>, \
    .nb_rshift = tensor_bitwise_shift_binop<BitwiseShiftOp::RShift>, \
    .nb_and = tensor_bitwise_binop<BinaryBitwiseOp::And>, \
    .nb_xor = tensor_bitwise_binop<BinaryBitwiseOp::Xor>, \
    .nb_or = tensor_bitwise_binop<BinaryBitwiseOp::Or>, \
    .nb_int = tensor_int, \
    .nb_float = tensor_float, \
    .nb_floor_divide = tensor_arith_binop<BinaryArithmeticOp::FloorDiv>, \
    .nb_true_divide = tensor_arith_binop<BinaryArithmeticOp::TrueDiv>, \
    .nb_index = tensor_index

static PyNumberMethods tensor_number_methods = {
    TENSOR_NUMBER_METHODS
};

static PyNumberMethods tile_number_methods = {
    TENSOR_NUMBER_METHODS,
    .nb_matrix_multiply = tile_nb_matrix_multiply,
};

#undef TENSOR_NUMBER_METHODS

PyTypeObject Tile::pytype = {
    .tp_name = "cuda.tile.Tile",
    .tp_basicsize = sizeof(Tile),
    .tp_dealloc = tensor_dealloc<Tile>,
    .tp_repr = tensor_repr,
    .tp_as_number = &tile_number_methods,
    .tp_as_mapping = &tile_mapping_methods,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = kDoc_Tile,
    .tp_richcompare = tensor_richcompare,
    .tp_methods = tile_methods,
    .tp_getset = tile_getset,
};

PyTypeObject Scalar::pytype = {
    .tp_name = "cuda.lang.Scalar",
    .tp_basicsize = sizeof(Scalar),
    .tp_dealloc = tensor_dealloc<Scalar>,
    .tp_repr = tensor_repr,
    .tp_as_number = &tensor_number_methods,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = kDoc_Scalar,
    .tp_richcompare = tensor_richcompare,
    .tp_getset = scalar_getset,
};

PyTypeObject Vector::pytype = {
    .tp_name = "cuda.lang.Vector",
    .tp_basicsize = sizeof(Vector),
    .tp_dealloc = tensor_dealloc<Vector>,
    .tp_repr = tensor_repr,
    .tp_as_number = &tensor_number_methods,
    .tp_as_sequence = &vector_sequence_methods,
    .tp_as_mapping = &vector_mapping_methods,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = kDoc_Vector,
    .tp_richcompare = tensor_richcompare,
    .tp_methods = vector_methods,
    .tp_getset = vector_getset,
};

PyTypeObject Pointer::pytype = {
    .tp_name = "cuda.lang.Pointer",
    .tp_basicsize = sizeof(Pointer),
    .tp_dealloc = tensor_dealloc<Pointer>,
    .tp_repr = tensor_repr,
    .tp_as_number = &tensor_number_methods,
    .tp_as_mapping = &pointer_mapping_methods,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = kDoc_Pointer,
    .tp_richcompare = tensor_richcompare,
    .tp_methods = pointer_methods,
    .tp_getset = pointer_getset,
};

PyTypeObject OpaqueValue::pytype = {
    .tp_name = "cuda.lang.OpaqueValue",
    .tp_basicsize = sizeof(OpaqueValue),
    .tp_dealloc = tensor_dealloc<OpaqueValue>,
    .tp_repr = tensor_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = kDoc_OpaqueValue,
    .tp_getset = opaque_value_getset,
};

static PyMethodDef value_functions[] = {
    {"_tile_reshape", py_tile_reshape, METH_VARARGS, nullptr},
    {"_tensorlike_astype", py_tensorlike_astype, METH_VARARGS, nullptr},
    {"_dtype_constructor", py_dtype_constructor, METH_VARARGS, nullptr},
    {"_tile_item", py_tile_item, METH_VARARGS, nullptr},
    {"_tile_getitem", py_tile_getitem, METH_VARARGS, nullptr},
    {"_vector_getitem", py_vector_getitem, METH_VARARGS, nullptr},
    {"_vector_with_item", py_vector_with_item, METH_VARARGS, nullptr},
    {"_tile_matmul", py_tile_matmul, METH_VARARGS, nullptr},
    {"_divmod_tensorlike", py_divmod_tensorlike, METH_VARARGS, nullptr},
    {"_tensorlike_negative", py_tensorlike_negative, METH_VARARGS, nullptr},
    {"_tensorlike_positive", py_tensorlike_positive, METH_VARARGS, nullptr},
    {"_tensorlike_invert", py_tensorlike_invert, METH_VARARGS, nullptr},
    {"_bitwise_shift_tensorlike", py_bitwise_shift_tensorlike, METH_VARARGS, nullptr},
    {"_binary_bitwise_tensorlike", py_binary_bitwise_tensorlike, METH_VARARGS, nullptr},
    {"_comparison_tensorlike", py_comparison_tensorlike, METH_VARARGS, nullptr},
    {"_binary_arithmetic_tensorlike", py_binary_arithmetic_tensorlike, METH_VARARGS, nullptr},
    {"_binary_wrapping_arithmetic_tensorlike", py_binary_wrapping_arithmetic_tensorlike,
     METH_VARARGS, nullptr},
    {"_make_symbolic_tensor", py_make_symbolic_tensor, METH_VARARGS, nullptr},
    {"_make_constant_0d_tensor", py_make_constant_0d_tensor, METH_VARARGS, nullptr},
    {"_tensor_to_ir_var", py_tensor_to_ir_var, METH_O, nullptr},
    {"encode_float", reinterpret_cast<PyCFunction>(py_encode_float), METH_FASTCALL, nullptr},
    {"decode_float", reinterpret_cast<PyCFunction>(py_decode_float), METH_FASTCALL, nullptr},
    {}
};

static Status add_type(PyObject* m, PyTypeObject* type, const char* name) {
    if (PyType_Ready(type) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, name, reinterpret_cast<PyObject*>(type)) < 0)
        return ErrorRaised;

    return OK;
}

Status value_module_init(PyObject* m) {
    INIT_STRING_IDENT(add);
    INIT_STRING_IDENT(sub);
    INIT_STRING_IDENT(mul);
    INIT_STRING_IDENT(wrapping_add);
    INIT_STRING_IDENT(wrapping_sub);
    INIT_STRING_IDENT(wrapping_mul);
    INIT_STRING_IDENT(truediv);
    INIT_STRING_IDENT(floordiv);
    INIT_STRING_IDENT(cdiv);
    INIT_STRING_IDENT(pow);
    INIT_STRING_IDENT(mod);
    INIT_STRING_IDENT(min);
    INIT_STRING_IDENT(max);
    INIT_STRING_IDENT(atan2);
    INIT_STRING_IDENT(eq);
    INIT_STRING_IDENT(ne);
    INIT_STRING_IDENT(lt);
    INIT_STRING_IDENT(le);
    INIT_STRING_IDENT(gt);
    INIT_STRING_IDENT(ge);
    INIT_STRING_IDENT(xor);
    INIT_STRING_IDENT(lshift);
    INIT_STRING_IDENT(rshift);
    INIT_STRING_IDENT(neg);
    INIT_STRING_IDENT(invert);
    INIT_STRING_IDENT(and_);
    INIT_STRING_IDENT(or_);

    if (!add_type(m, &Tile::pytype, "Tile")) return ErrorRaised;
    if (!add_type(m, &Scalar::pytype, "Scalar")) return ErrorRaised;
    if (!add_type(m, &Vector::pytype, "Vector")) return ErrorRaised;
    if (!add_type(m, &Pointer::pytype, "Pointer")) return ErrorRaised;
    if (!add_type(m, &OpaqueValue::pytype, "OpaqueValue")) return ErrorRaised;

    if (PyModule_AddFunctions(m, value_functions) < 0) return ErrorRaised;

    return OK;
}


