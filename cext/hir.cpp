// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "hir.h"
#include "value.h"
#include "growable_stack.h"
#include "temporary_array.h"

#include <frameobject.h>  // For PyFrame_Check on Python 3.10
#include <optional>


#define ENUM_ENTRY(name) name,

#define FOREACH_HIR_OPCODE(X) \
    X(GenericCall) \
    X(LoadLocal) \
    X(StoreLocal) \
    X(LoadGlobal) \
    X(LoadUndefinedVariable) \
    X(BuildFormattedString) \
    X(BuildTuple) \
    X(Unpack) \
    X(MakeClosure) \
    X(StaticEval) \
    X(StaticAssert) \
    X(GeneratorYield) \
    X(IfElse) \
    X(TupleCompIf) \
    X(Loop) \
    X(StaticForeach) \
    X(EndBranch) \
    X(Continue) \
    X(Break) \
    X(Return) \
    X(FunctionResult)


enum class HirOpcode : uint8_t {
    FOREACH_HIR_OPCODE(ENUM_ENTRY)
    BeginBlock,
    EndBlock,
};

template <HirOpcode Opcode> constexpr const char* opcode_str();

#define OPCODE_STR_SPEC(name) \
    template <> constexpr const char* opcode_str<HirOpcode::name>() { return #name; }
FOREACH_HIR_OPCODE(OPCODE_STR_SPEC)
#undef OPCODE_STR_SPEC


#define FOREACH_HIR_COROUTINE_KIND(X) \
    X(EvalBlock) \
    X(PythonCoroutine)

enum class HirCoroutineKind : uint8_t {
    FOREACH_HIR_COROUTINE_KIND(ENUM_ENTRY)
};

#undef ENUM_ENTRY

struct InstructionPointer {
    uint32_t offset;  // offset into the instruction data array, in words
    uint32_t index;  // 0-based index of the instruction
};

union HirWord {
    constexpr static size_t kHeaderDataBits = 56;

    struct {
        uint64_t opcode : 8;  // HirOpcode (using uint64_t for MSVC compatibility)
        uint64_t data : kHeaderDataBits;
    } header;
    PyObject* pyobject;
    uint64_t u64;
    InstructionPointer iptr;
};

static_assert(sizeof(HirWord) == sizeof(uint64_t));

static HirOpcode header_opcode(const HirWord& word) {
    return static_cast<HirOpcode>(word.header.opcode);
}

static HirWord make_header(HirOpcode opcode, uint64_t data) {
    return HirWord{.header = {static_cast<uint64_t>(opcode), data}};
}

template <HirOpcode> struct OpcodeImpl;

template <typename T> struct AccessorBase {
    HirWord* words;
    T attr;
};

template <typename T> struct Accessor;

struct BitmapAttr {
    size_t num_bits;

    static constexpr size_t kBitsPerWord = sizeof(uint64_t) * 8;
    constexpr size_t get_num_words() const { return (num_bits + kBitsPerWord - 1) / kBitsPerWord; }
};

template <> struct Accessor<BitmapAttr> : AccessorBase<BitmapAttr> {
    bool operator[] (size_t bit) const {
        return (words[bit / BitmapAttr::kBitsPerWord].u64 >> (bit % BitmapAttr::kBitsPerWord)) & 1;
    }

    void set(size_t bit) const {
        words[bit / BitmapAttr::kBitsPerWord].u64 |= 1ull << (bit % BitmapAttr::kBitsPerWord);
    }

    bool is_zero() const {
        size_t num_words = attr.get_num_words();
        for (size_t i = 0; i < num_words; ++i) {
            if (words[i].u64) return false;
        }
        return true;
    }

    void put(HirWord* dst) const {
        size_t num_words = attr.get_num_words();
        for (size_t i = 0; i < num_words; ++i)
            dst[i].u64 = words[i].u64;
    }

    void print(StringBuilder* sb) const {
        for (size_t i = 0; i < attr.num_bits; ++i)
            sb->append((*this)[i] ? "1" : "0");
    }

    void destroy() const {}  // nothing to do, just plain bits
};

struct StringSeqAttr {
    size_t num_strings;

    constexpr size_t get_num_words() const { return num_strings; }
};


template <> struct Accessor<StringSeqAttr> : AccessorBase<StringSeqAttr> {
    PyObject* operator[] (size_t i) const {
        return words[i].pyobject;
    }

    void set(size_t i, PyPtr&& obj) const {
        words[i].pyobject = obj.release();
    }

    void put(HirWord* dst) const {
        for (size_t i = 0; i < attr.num_strings; ++i)
            dst[i].pyobject = Py_XNewRef(words[i].pyobject);
    }

    void print(StringBuilder* sb) const {
        sb->append("[");
        for (size_t i = 0; i < attr.num_strings; ++i) {
            if (i > 0) sb->append(", ");
            sb->append((*this)[i]);
        }
        sb->append("]");
    }

    void destroy() const {
        for (size_t i = 0; i < attr.num_strings; ++i)
            Py_XDECREF(words[i].pyobject);
    }
};


struct PyobjectAttr {
    constexpr size_t get_num_words() const { return 1; }
};


template <> struct Accessor<PyobjectAttr> : AccessorBase<PyobjectAttr> {
    PyObject* get() const {
        return words->pyobject;
    }

    void set(PyPtr&& obj) const {
        words->pyobject = obj.release();
    }

    void put(HirWord* dst) const {
        dst->pyobject = Py_XNewRef(words->pyobject);
    }

    void print(StringBuilder* sb) const {
        sb->append(get());
    }

    void destroy() const {
        Py_XDECREF(words->pyobject);
    }
};


struct LabelAttr {
    constexpr size_t get_num_words() const { return 1; }
};


template <> struct Accessor<LabelAttr> : AccessorBase<LabelAttr> {
    InstructionPointer get() const {
        return words->iptr;
    }

    void set(InstructionPointer iptr) const {
        words->iptr = iptr;
    }

    void put(HirWord* dst) const {
        dst->iptr = words->iptr;
    }

    void print(StringBuilder* sb) const {
        sb->append_many("^", words->iptr.index);
    }

    void destroy() const {}  // nothing to do, just plain data
};


template <typename T>
struct OwningAccessor {
    Accessor<T> accessor;

    explicit OwningAccessor<T>(const Accessor<T>& accessor) : accessor(accessor) {}

    OwningAccessor(const OwningAccessor&) = delete;
    void operator=(const OwningAccessor&) = delete;

    ~OwningAccessor() {
        accessor.destroy();
    }

    const Accessor<T>& operator*() const { return accessor; }
    const Accessor<T>* operator->() const { return &accessor; }
};

template <typename F, size_t... I>
constexpr void static_for_impl(F&& func, std::index_sequence<I...>) {
    (func(std::integral_constant<size_t, I>{}), ...);
}

template <size_t N, typename F>
constexpr void static_for(F&& func) {
    static_for_impl(func, std::make_index_sequence<N>());
}


template <bool RequiresLoc, typename NumOperands, typename HeaderMeta, typename... Attr>
struct OpcodeLayout {
    using HeaderMetaTuple = HeaderMeta;

    static constexpr bool kRequiresLoc = RequiresLoc;

    NumOperands num_operands;
    HeaderMetaTuple header_meta;
    std::tuple<Attr...> attributes;

    using AccessorTuple = std::tuple<Accessor<Attr>...>;

    constexpr size_t attribute_words() const {
        return std::apply(
            [] (auto&&... a) { return (a.get_num_words() + ... + 0); },
            attributes);
    }

    constexpr AccessorTuple parse_attributes(HirWord** cursor) {
        return std::apply(
            [cursor] (auto&&... a) {
                // Braced init guarantees left-to-right evaluation
                return AccessorTuple{Accessor<std::remove_reference_t<decltype(a)>>{
                    std::exchange(*cursor, *cursor + a.get_num_words()), a}...};
            }, attributes);
    }

    void put_attributes(HirWord* dst, Accessor<Attr>... attrs) {
        _put_attributes_helper(
                dst, std::make_tuple(attrs...), std::make_index_sequence<sizeof...(Attr)>());
    }

    template <size_t... Idx>
    void _put_attributes_helper(HirWord* dst, const AccessorTuple& src,
                                std::index_sequence<Idx...>) {
        ((std::get<Idx>(src).put(dst), dst += std::get<Idx>(attributes).get_num_words()), ...);
    }
};



template <typename T>
struct is_u32_constant : std::false_type {};

template <uint32_t N>
struct is_u32_constant<std::integral_constant<uint32_t, N>> : std::true_type {};

template <typename T>
constexpr bool is_u32_constant_v = is_u32_constant<T>::value;


template <bool RequiresLoc = false, typename NumOperands, typename... HeaderMeta,
          typename... Attr>
static auto make_opcode_layout(NumOperands&& num_operands,
                               std::tuple<HeaderMeta...> header_meta,
                               std::tuple<Attr...> attributes) {
    using N = std::remove_reference_t<NumOperands>;
    static_assert(std::is_same_v<N, uint32_t> || is_u32_constant_v<N>);
    return OpcodeLayout<RequiresLoc, N, std::tuple<HeaderMeta...>, Attr...>{
            num_operands, std::move(header_meta), std::move(attributes)};
}

template <HirOpcode Opcode>
constexpr bool kOpcodeRequiresLoc
        = decltype(OpcodeImpl<Opcode>::parse_layout(HirWord{}))::kRequiresLoc;


template <typename Attr>
static Accessor<Attr> allocate_attribute(GrowableStack* stack, const Attr& attr) {
    HirWord* data = stack->allocate_zeroed<HirWord>(attr.get_num_words());
    return Accessor<Attr>{data, attr};
}

template <HirOpcode Opcode, typename... Args>
static auto allocate_operands_and_attributes(GrowableStack* stack, Args&&... args) {
    auto layout = OpcodeImpl<Opcode>::make_layout(std::forward<Args>(args)...);

    BorrowedValue* operands = stack->allocate_zeroed<BorrowedValue>(layout.num_operands);
    return std::apply(
        [&] (auto&&... attr) {
            return std::tuple<BorrowedValue*, uint32_t,
                              OwningAccessor<std::remove_reference_t<decltype(attr)>>...>(
                    operands, layout.num_operands, allocate_attribute(stack, attr)...);
        }, layout.attributes);
}

template <bool PartialEval> struct HirEvalContext;


static void print_operand(BorrowedValue value, StringBuilder* sb) {
    if (value.is_hir_value_id()) {
        sb->append_many("%", value.as_hir_value_id());
    } else {
        // TODO: nicer printing of specific values
        value.append_repr_to_builder(sb);
    }
}

template <HirOpcode Opcode> struct OpcodeDispatchKey{};


template <HirOpcode Opcode>
struct ParsedInstruction {
    static constexpr HirOpcode kOpcode = Opcode;

    using Layout = decltype(OpcodeImpl<Opcode>::parse_layout(HirWord{}));
    using HeaderMetaTuple = typename Layout::HeaderMetaTuple;
    using AttrTuple = typename Layout::AccessorTuple;
    using NumOperands = decltype(Layout::num_operands);

    const HirWord* operands;
    NumOperands num_operands;  // uint32_t or std::integral_constant<uint32_t, N>
    HeaderMetaTuple header_meta;
    AttrTuple attributes;
};

template <> struct ParsedInstruction<HirOpcode::BeginBlock> {};
template <> struct ParsedInstruction<HirOpcode::EndBlock> {};


template <HirOpcode Opcode>
static void print_instruction(const ParsedInstruction<Opcode>& instruction,
                              uint32_t instruction_index,
                              StringBuilder* sb) {
    if constexpr (Opcode == HirOpcode::BeginBlock) {
        sb->append_many("^", instruction_index + 1, "(%", instruction_index, ") {");
    } else if constexpr (Opcode == HirOpcode::EndBlock) {
        sb->append("}");
    } else {
        sb->append_many("%", instruction_index, " = ", opcode_str<Opcode>());
        std::apply(
            [&] (auto&&... m) { (sb->append_many(" ", m), ...); },
            instruction.header_meta
        );
        std::apply(
            [&] (auto&&... a) { ((sb->append(" "), a.print(sb)), ...); },
            instruction.attributes
        );
        sb->append(" (");
        for (uint32_t i = 0; i < instruction.num_operands; ++i) {
            if (i > 0) sb->append(", ");
            print_operand(BorrowedValue::from_raw_data(instruction.operands[i].u64), sb);
        }
        sb->append(")");
    }
}

template <HirOpcode Opcode>
static void destroy_instruction(const ParsedInstruction<Opcode>& instruction) {
    if constexpr (Opcode != HirOpcode::BeginBlock && Opcode != HirOpcode::EndBlock) {
        // Destroy the attributes
        std::apply([&] (auto&&... a) { (a.destroy(), ...); }, instruction.attributes);

        // Destroy the operands
        for (uint32_t i = 0; i < instruction.num_operands; ++i) {
            // Call the Value destructor
            BorrowedValue::from_raw_data(instruction.operands[i].u64).steal();
        }
    }
}

template <HirOpcode Opcode>
static ParsedInstruction<Opcode> parse_instruction(HirWord** cursor) {
    auto layout = OpcodeImpl<Opcode>::parse_layout(**cursor);
    ++*cursor;  // skip the header
    auto attrs_tuple = layout.parse_attributes(cursor);
    HirWord* operand_words = *cursor;
    *cursor += layout.num_operands;
    return {operand_words, layout.num_operands, layout.header_meta, std::move(attrs_tuple)};
}


// Utility class for extracting the first items of known types from the parameter pack.
template <typename Tuple> struct TakeFirst;

template <typename... Taken>
struct TakeFirst<std::tuple<Taken...>> {
    template <typename... Tail>
    std::tuple<Taken...> operator()(const Taken&... taken, Tail&&...) {
        return std::tuple<Taken...>(taken...);
    }
};

// Utility class for removing the first N items of known types from the parameter pack.
template <typename... Dropped>
struct DropFirst {
    template <typename F, typename... Tail>
    auto operator()(F&& func, const Dropped&..., Tail&&... tail) {
        return func(std::forward<Tail>(tail)...);
    }
};

template <typename T, size_t Idx> using RepeatedType = T;

template <typename T, size_t... Idx>
constexpr auto make_drop_first_homogenous(std::index_sequence<Idx...>) {
    return DropFirst<RepeatedType<T, Idx>...>{};
}


template <typename NumOperands>
auto make_operand_dropper() {
    if constexpr (std::is_same_v<NumOperands, uint32_t>) {
        // Variadic operands: drop the operands pointer, keep the count
        return DropFirst<const BorrowedValue*>{};
    } else {
        // Fixed operands: drop all BorrowedValue arguments
        static_assert(is_u32_constant_v<NumOperands>);
        return make_drop_first_homogenous<BorrowedValue>(
                std::make_index_sequence<NumOperands::value>());
    }
}

template <typename NumOperands>
auto make_operand_and_count_dropper() {
    if constexpr (std::is_same_v<NumOperands, uint32_t>) {
        // Variadic operands: drop the operands pointer and the count
        return DropFirst<const BorrowedValue*, uint32_t>{};
    } else {
        return make_operand_dropper<NumOperands>();
    }
}

template <typename... DroppedOperands, typename... HeaderMeta>
constexpr auto make_operand_and_header_meta_dropper(
        const DropFirst<DroppedOperands...>& operand_dropper,
        const std::tuple<HeaderMeta...>& header_meta) {
    return DropFirst<DroppedOperands..., HeaderMeta...>{};
}

template <typename... Args>
static std::pair<const BorrowedValue*, uint32_t> get_variadic_operands(
        const BorrowedValue* operands, uint32_t num_operands, Args&&...) {
    return {operands, num_operands};
}

struct HirProgram {
    Vec<HirWord> data;
    Vec<PyPtr> locations;  // location of each instruction
    PyPtr loc;

    ~HirProgram();

    uint32_t instruction_count() const {
        return static_cast<uint32_t>(locations.size());
    }

    InstructionPointer end_iptr() const {
        return {static_cast<uint32_t>(data.size()), instruction_count()};
    }

    // Returns the header of the BeginBlock instruction that starts the given block.
    // The BeginBlock instruction consists of the header and the end label.
    HirWord& begin_block_header(InstructionPointer block_start) {
        CHECK(block_start.offset >= 2);
        HirWord& header = data[block_start.offset - 2];
        CHECK(header_opcode(header) == HirOpcode::BeginBlock);
        return header;
    }

    bool get_block_last_opcode(InstructionPointer block_start, HirOpcode* out) {
        InstructionPointer end = (&begin_block_header(block_start))[1].iptr;
        if (!end.offset) return false;
        CHECK(end.offset >= block_start.offset + 1);
        const HirWord& end_header = data[end.offset - 1];
        CHECK(header_opcode(end_header) == HirOpcode::EndBlock);
        *out = static_cast<HirOpcode>(end_header.header.data);
        return true;
    }

    static PyTypeObject pytype;
};


struct HirValue {
    Value value;
    PyPtr loc;
    static PyTypeObject pytype;
};

static Result<BorrowedValue> parse_operand(PyObject* obj);

static PyObject* HirValue_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    if (PyTuple_GET_SIZE(args) != 2 || (kwargs && PyDict_GET_SIZE(kwargs) != 0)) {
        raise(PyExc_TypeError, "HirValue(value, loc) expected 2 positional arguments");
        return nullptr;
    }

    Result<BorrowedValue> value_res = parse_operand(PyTuple_GET_ITEM(args, 0));
    if (!value_res.is_ok()) return nullptr;

    PyPtr ret = py_create_object<HirValue>(type);
    if (!ret) return nullptr;
    HirValue& hir_value = py_unwrap<HirValue>(ret);
    hir_value.value = value_res->to_owned();
    hir_value.loc = newref(PyTuple_GET_ITEM(args, 1));
    return ret.release();
}

static PyObject* HirValue_get_loc(PyObject* self, void*) {
    return Py_NewRef(py_unwrap<HirValue>(self).loc.get());
}

static PyGetSetDef HirValue_getset[] = {
    {"loc", HirValue_get_loc, nullptr, nullptr, nullptr},
    {}
};

PyTypeObject HirValue::pytype = {
    .tp_name = "cuda.tile._cext.HirValue",
    .tp_basicsize = sizeof(PythonWrapper<HirValue>),
    .tp_dealloc = pywrapper_dealloc<HirValue>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = HirValue_getset,
    .tp_new = HirValue_new,
};

static PyPtr wrap_value(Value value, PyPtr loc) {
    return py_create_object_with_args<HirValue>(HirValue{std::move(value), std::move(loc)});
}


struct HirLabel {
    PyPtr program;  // HirProgram
    InstructionPointer iptr;

    bool operator== (const HirLabel& that) const {
        // Don't need to check the instruction offset
        return program == that.program && iptr.index == that.iptr.index;
    }

    static PyTypeObject pytype;
};

// Returns the value ID of the block's BeginBlock opcode,
// which holds the block parameter (e.g. the loop induction variable)
static PyObject* HirLabel_block_parameter(PyObject* self, PyObject* loc) {
    const HirLabel& label = py_unwrap<HirLabel>(self);
    if (!label.iptr.index) {
        raise(PyExc_ValueError, "Label 0 does not point to start of a block");
        return nullptr;
    }

    Value value = BorrowedValue::hir_value_id(label.iptr.index - 1).to_owned();
    return wrap_value(std::move(value), newref(loc)).release();
}

// Returns the number of Continue, Break and Return opcodes inside the block,
// including those nested in IfElse branches
static PyObject* HirLabel_num_nested_jumps(PyObject* self, PyObject*) {
    const HirLabel& label = py_unwrap<HirLabel>(self);
    if (!label.iptr.index) {
        raise(PyExc_ValueError, "Label 0 does not point to start of a block");
        return nullptr;
    }

    HirProgram& prog = py_unwrap<HirProgram>(label.program);
    return PyLong_FromUnsignedLongLong(prog.begin_block_header(label.iptr).header.data);
}

static PyObject* HirLabel_ends_with_break(PyObject* self, PyObject*) {
    const HirLabel& label = py_unwrap<HirLabel>(self);
    if (!label.iptr.index) {
        raise(PyExc_ValueError, "Label 0 does not point to start of a block");
        return nullptr;
    }

    HirProgram& prog = py_unwrap<HirProgram>(label.program);
    HirOpcode last_opcode;
    if (!prog.get_block_last_opcode(label.iptr, &last_opcode)) {
        raise(PyExc_ValueError, "Block has not been closed yet");
        return nullptr;
    }
    return PyBool_FromLong(last_opcode == HirOpcode::Break);
}

static Py_hash_t HirLabel_hash(PyObject* self) {
    // Don't bother hashing the program since in each particular dictionary,
    // we will typically only have labels from the same program.
    return py_unwrap<HirLabel>(self).iptr.index;
}

static PyMethodDef HirLabel_methods[] = {
    {"block_parameter", HirLabel_block_parameter, METH_O, nullptr},
    {"num_nested_jumps", HirLabel_num_nested_jumps, METH_NOARGS, nullptr},
    {"ends_with_break", HirLabel_ends_with_break, METH_NOARGS, nullptr},
    {}
};

static PyObject* HirLabel_get_loc(PyObject* self, void*) {
    const HirLabel& label = py_unwrap<HirLabel>(self);
    const HirProgram& prog = py_unwrap<HirProgram>(label.program);
    if (!label.iptr.index)
        return Py_NewRef(prog.loc.get());

    // The location of the BeginBlock instruction that immediately precedes the block start
    return Py_NewRef(prog.locations[label.iptr.index - 1].get());
}

static PyGetSetDef HirLabel_getset[] = {
    {"loc", HirLabel_get_loc, nullptr, nullptr, nullptr},
    {}
};

PyTypeObject HirLabel::pytype = {
    .tp_name = "cuda.tile._cext.HirLabel",
    .tp_basicsize = sizeof(PythonWrapper<HirLabel>),
    .tp_dealloc = pywrapper_dealloc<HirLabel>,
    .tp_hash = HirLabel_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = pywrapper_richcompare_via_operator_equals<HirLabel>,
    .tp_methods = HirLabel_methods,
    .tp_getset = HirLabel_getset,
};

static PyPtr hir_label_new(PyPtr program, InstructionPointer iptr) {
    return py_create_object_with_args<HirLabel>(HirLabel{std::move(program), iptr});
}


static void print_block(HirProgram& program, InstructionPointer start, size_t indent,
                        std::optional<uint32_t> marked_instruction_id, StringBuilder* sb);


static PyObject* HirProgram_repr(PyObject* self) {
    StringBuilder sb;
    print_block(py_unwrap<HirProgram>(self), {}, 0, std::nullopt, &sb);
    return sb.build().release();
}

static PyObject* HirProgram_block_str(PyObject* self, PyObject* py_label) {
    if (Py_TYPE(py_label) != &HirLabel::pytype) {
        raise(PyExc_TypeError, "label must be a HirLabel, not ", Py_TYPE(py_label)->tp_name);
        return nullptr;
    }
    const HirLabel& label = py_unwrap<HirLabel>(py_label);
    if (self != label.program.get()) {
        raise(PyExc_ValueError, "Label from wrong program");
        return nullptr;
    }

    HirProgram& p = py_unwrap<HirProgram>(self);
    InstructionPointer start = label.iptr;

    StringBuilder sb;
    if (start.index == 0) {
        // The whole program: there is no enclosing BeginBlock
        print_block(p, {}, 0, std::nullopt, &sb);
    } else {
        sb.append_many("^", start.index, "(%", start.index - 1, ") {\n");
        print_block(p, start, 1, std::nullopt, &sb);
        sb.append("}\n");
    }
    return sb.build().release();
}

// Returns the label of the whole program, which is not a nested block
static PyObject* HirProgram_start_label(PyObject* self, PyObject*) {
    return hir_label_new(newref(self), InstructionPointer{0, 0}).release();
}

static PyMethodDef HirProgram_methods[] = {
    {"block_str", HirProgram_block_str, METH_O, nullptr},
    {"start_label", HirProgram_start_label, METH_NOARGS, nullptr},
    {}
};

PyTypeObject HirProgram::pytype = {
    .tp_name = "cuda.tile._cext.HirProgram",
    .tp_basicsize = sizeof(PythonWrapper<HirProgram>),
    .tp_dealloc = pywrapper_dealloc<HirProgram>,
    .tp_repr = HirProgram_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = HirProgram_methods,
};


struct HirRunner;

// The value passed between coroutines: the result of the finished coroutine,
// or the value to send to the resumed one. If `exc` is set, it is thrown instead.
struct CoroutineValue {
    PyPtr value;
    SavedException exc;

    void set_success(PyPtr new_value) {
        CHECK(new_value);
        value = std::move(new_value);
        exc.clear();
    }

    void save_raised_exception() {
        exc = ::save_raised_exception();
        exc.normalize();
        value = newref(Py_None);
    }
};

template <HirCoroutineKind Kind>
struct HirCoroutineImpl;


struct HirCoroutine {
    GrowableStack::Watermark watermark;
    HirCoroutine* prev;
    HirCoroutineKind kind;
    void* impl;  // Pointer to HirCoroutineImpl<kind>

    HirCoroutine(GrowableStack::Watermark watermark,
                 HirCoroutine* prev,
                 HirCoroutineKind kind,
                 void* impl)
        : watermark(watermark), prev(prev), kind(kind), impl(impl)
    { }

    template <typename F>
    auto dispatch(F&& func) {
        switch (kind) {
        #define DISPATCH_CORO(name) \
            case HirCoroutineKind::name: \
                return func(static_cast<HirCoroutineImpl<HirCoroutineKind::name>*>(impl));
        FOREACH_HIR_COROUTINE_KIND(DISPATCH_CORO)
        #undef DISPATCH_CORO
        }
        CHECK_UNREACHABLE;
    }
};


struct HirFrame {
    PyObject_VAR_HEAD
    PyPtr program;  // HirProgram
    PyPtr call_site_loc;  // Loc or nullptr
    PyPtr function_desc;  // FunctionDesc

    // Returns a null pointer on error
    static RefPtr<HirFrame> create(PyPtr program, PyPtr call_site_loc, PyPtr function_desc) {
        CHECK(Py_TYPE(program.get()) == &HirProgram::pytype);
        size_t size = py_unwrap<HirProgram>(program).instruction_count();
        PyObject* obj = pytype.tp_alloc(&pytype, size);
        if (!obj) return {};

        HirFrame* ret = reinterpret_cast<HirFrame*>(obj);
        new (&ret->program) PyPtr(std::move(program));
        new (&ret->call_site_loc) PyPtr(std::move(call_site_loc));
        new (&ret->function_desc) PyPtr(std::move(function_desc));
        Value* values = ret->value_map();
        for (size_t i = 0; i < size; ++i)
            new (&values[i]) Value();
        return steal(ret);
    }

    HirFrame(const HirFrame&) = delete;
    void operator=(const HirFrame&) = delete;

    ~HirFrame() {
        Value* values = value_map();
        for (size_t i = 0; i < size(); ++i)
            values[i].~Value();
    }

    size_t size() const {
        return ob_base.ob_size;
    }

    Value* value_map() {
        return reinterpret_cast<Value*>(this + 1);
    }

    static PyTypeObject pytype;
};

static_assert(sizeof(HirFrame) % alignof(Value) == 0);

static inline void reference_add(HirFrame& frame) {
    Py_INCREF(reinterpret_cast<PyObject*>(&frame));
}

static inline void reference_remove(HirFrame& frame) {
    Py_DECREF(reinterpret_cast<PyObject*>(&frame));
}

static void HirFrame_dealloc(PyObject* self) {
    auto free_func = Py_TYPE(self)->tp_free;
    reinterpret_cast<HirFrame*>(self)->~HirFrame();
    free_func(self);
}

static PyObject* HirFrame_new(PyTypeObject*, PyObject* args, PyObject* kwargs) {
    if (PyTuple_GET_SIZE(args) != 3 || (kwargs && PyDict_GET_SIZE(kwargs) != 0)) {
        raise(PyExc_TypeError,
              "HirFrame(program, call_site_loc, function_desc) expected 3 positional arguments");
        return nullptr;
    }

    PyObject* program = PyTuple_GET_ITEM(args, 0);
    if (Py_TYPE(program) != &HirProgram::pytype) {
        raise(PyExc_TypeError, "program must be a HirProgram, not ", Py_TYPE(program)->tp_name);
        return nullptr;
    }

    PyObject* call_site_loc = PyTuple_GET_ITEM(args, 1);
    PyObject* function_desc = PyTuple_GET_ITEM(args, 2);
    RefPtr<HirFrame> frame = HirFrame::create(
            newref(program),
            call_site_loc == Py_None ? PyPtr() : newref(call_site_loc),
            newref(function_desc));
    return reinterpret_cast<PyObject*>(frame.release());
}

static PyObject* HirFrame_get_call_site_loc(PyObject* self, void*) {
    PyObject* loc = reinterpret_cast<HirFrame*>(self)->call_site_loc.get();
    return Py_NewRef(loc ? loc : Py_None);
}

static PyObject* HirFrame_get_function_desc(PyObject* self, void*) {
    return Py_NewRef(reinterpret_cast<HirFrame*>(self)->function_desc.get());
}

static PyGetSetDef HirFrame_getset[] = {
    {"call_site_loc", HirFrame_get_call_site_loc, nullptr, nullptr, nullptr},
    {"function_desc", HirFrame_get_function_desc, nullptr, nullptr, nullptr},
    {}
};

PyTypeObject HirFrame::pytype = {
    .tp_name = "cuda.tile._cext.HirFrame",
    .tp_basicsize = sizeof(HirFrame),
    .tp_itemsize = sizeof(Value),
    .tp_dealloc = HirFrame_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = HirFrame_getset,
    .tp_new = HirFrame_new,
};


struct EvalBlockAwaitable {
    enum State : uint8_t {
        kInitial,
        kSuspended,
        kDone,
    };

    RefPtr<HirFrame> frame;
    InstructionPointer iptr;
    Value block_parameter;  // undefined if not given
    State state = kInitial;

    static PyTypeObject pytype;
};

static PySendResult EvalBlockAwaitable_am_send(PyObject* self, PyObject* arg, PyObject** result) {
    EvalBlockAwaitable& awaitable = py_unwrap<EvalBlockAwaitable>(self);
    switch (awaitable.state) {
    case EvalBlockAwaitable::kInitial:
        awaitable.state = EvalBlockAwaitable::kSuspended;
        *result = Py_NewRef(self);
        return PYGEN_NEXT;
    case EvalBlockAwaitable::kSuspended:
        awaitable.state = EvalBlockAwaitable::kDone;
        *result = Py_NewRef(arg);
        return PYGEN_RETURN;
    case EvalBlockAwaitable::kDone:
        raise(PyExc_RuntimeError, "EvalBlockAwaitable cannot be awaited more than once");
        *result = nullptr;
        return PYGEN_ERROR;
    }
    CHECK_UNREACHABLE;
}

static PyObject* EvalBlockAwaitable_send(PyObject* self, PyObject* arg) {
    PyObject* result;
    switch (EvalBlockAwaitable_am_send(self, arg, &result)) {
    case PYGEN_NEXT:
        return result;
    case PYGEN_RETURN: {
        PyPtr value = steal(result);
        PyPtr exc = steal(PyObject_CallOneArg(PyExc_StopIteration, value.get()));
        if (exc) PyErr_SetObject(PyExc_StopIteration, exc.get());
        return nullptr;
    }
    case PYGEN_ERROR:
        return nullptr;
    }
    CHECK_UNREACHABLE;
}

static PyObject* EvalBlockAwaitable_iternext(PyObject* self) {
    return EvalBlockAwaitable_send(self, Py_None);
}

static PyObject* EvalBlockAwaitable_iter_and_am_await(PyObject* self) {
    return Py_NewRef(self);
}

static PyAsyncMethods EvalBlockAwaitable_async_methods = {
    .am_await = EvalBlockAwaitable_iter_and_am_await,
    .am_send = EvalBlockAwaitable_am_send,
};

static PyMethodDef EvalBlockAwaitable_methods[] = {
    {"send", EvalBlockAwaitable_send, METH_O, nullptr},
    {}
};

PyTypeObject EvalBlockAwaitable::pytype = {
    .tp_name = "cuda.tile._cext.EvalBlockAwaitable",
    .tp_basicsize = sizeof(PythonWrapper<EvalBlockAwaitable>),
    .tp_dealloc = pywrapper_dealloc<EvalBlockAwaitable>,
    .tp_as_async = &EvalBlockAwaitable_async_methods,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = EvalBlockAwaitable_iter_and_am_await,
    .tp_iternext = EvalBlockAwaitable_iternext,
    .tp_methods = EvalBlockAwaitable_methods,
};


struct HirRunner {
    GrowableStack stack;
    HirCoroutine* top = nullptr;
    bool dump_hir_on_error = false;

    HirRunner() = default;

    HirRunner(const HirRunner&) = delete;
    void operator=(const HirRunner&) = delete;

    ~HirRunner() {
        clear();
    }

    void clear() {
        while (!empty()) {
            top->dispatch([] (auto impl) { std::destroy_at(impl); });
            pop();
        }
    }

    // Pop the frame without destroying it
    void pop() {
        HirCoroutine* prev = top->prev;
        stack.reset_to(top->watermark);
        top = prev;
    }

    bool empty() const {
        return !top;
    }

    template <HirCoroutineKind Kind, typename... Args>
    HirCoroutineImpl<Kind>* push_coroutine(Args&&... args) {
        GrowableStack::Watermark watermark = stack.current_watermark();
        HirCoroutineImpl<Kind>* impl = stack.emplace<HirCoroutineImpl<Kind>>(
                std::forward<Args>(args)...);
        HirCoroutine* new_coro = stack.emplace<HirCoroutine>(watermark, top, Kind, impl);
        top = new_coro;
        return impl;
    }

    void push_eval_block(RefPtr<HirFrame> frame, InstructionPointer iptr) {
        push_coroutine<HirCoroutineKind::EvalBlock>(std::move(frame), iptr, dump_hir_on_error);
    }

    Status push_continuation(PyPtr cont, bool entry) {
        if (Py_TYPE(cont.get()) == &EvalBlockAwaitable::pytype) {
            EvalBlockAwaitable& e = py_unwrap<EvalBlockAwaitable>(cont.get());
            switch (e.state) {
            case EvalBlockAwaitable::kInitial:
                if (!entry)
                    return raise(PyExc_RuntimeError,
                                 "EvalBlockAwaitable must be awaited, not yielded directly");
                // Nothing will send the result to the awaitable, so mark it as done right away
                e.state = EvalBlockAwaitable::kDone;
                break;
            case EvalBlockAwaitable::kSuspended:
                if (entry)
                    return raise(PyExc_RuntimeError, "EvalBlockAwaitable is already being awaited");
                // Yielded by an awaiting coroutine, which will be resumed with the result
                break;
            case EvalBlockAwaitable::kDone:
                return raise(PyExc_RuntimeError,
                             "EvalBlockAwaitable cannot be awaited more than once");
            }
            // Store the block parameter (e.g. the loop induction variable) in the value slot
            // of the block's BeginBlock opcode
            if (!e.block_parameter.borrow().is_undefined()) {
                CHECK(e.iptr.index);
                e.frame->value_map()[e.iptr.index - 1] = std::move(e.block_parameter);
            }
            push_eval_block(e.frame, e.iptr);
            return OK;
        } else if (PyCoro_CheckExact(cont.get())) {
            push_coroutine<HirCoroutineKind::PythonCoroutine>(std::move(cont));
            return OK;
        } else {
            return raise(PyExc_TypeError,
                         "awaitable must be an EvalBlockAwaitable or a coroutine, not ",
                         Py_TYPE(cont.get())->tp_name);
        }
    }
};

static Result<Value> value_from_ir_var(PyPtr&& var);

// Returns true if the object is an instance of the given class from cuda.tile._ir.type
static Result<bool> is_ir_type(PyObject* obj, LazyImporter* importer, const char* class_name) {
    PyObject* cls = importer->get("cuda.tile._ir.type", class_name);
    if (!cls) return ErrorRaised;
    CHECK(PyType_Check(cls));
    return bool(PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(cls)));
}

// Converts the items of an aggregate Var (e.g. the Vars of a tuple) to Values
static Status aggregate_items_from_ir_var(PyObject* var, Value* out, size_t num_items) {
    RecursionGuard guard(" while converting an aggregate ir.Var to a Value");
    if (!guard) return ErrorRaised;
    PyPtr agg = steal(PyObject_CallMethod(var, "get_aggregate", nullptr));
    if (!agg) return ErrorRaised;
    PyPtr items = steal(PyObject_CallMethod(agg.get(), "as_tuple", nullptr));
    if (!items) return ErrorRaised;
    CHECK(PyTuple_Check(items.get()));
    CHECK(static_cast<size_t>(PyTuple_GET_SIZE(items.get())) == num_items);
    for (size_t i = 0; i < num_items; ++i) {
        Result<Value> r = value_from_ir_var(newref(PyTuple_GET_ITEM(items.get(), i)));
        if (!r.is_ok()) return ErrorRaised;
        out[i] = std::move(*r);
    }
    return OK;
}

static Result<Value> value_from_ir_var(PyPtr&& var) {
    if (var.get() == Py_None)
        return Value::none();

    static LazyImporter var_ty_importer;
    PyObject* var_ty = var_ty_importer.get("cuda.tile._ir.ir", "Var");
    if (!var_ty) return ErrorRaised;

    PyTypeObject* ty = Py_TYPE(var.get());
    if (reinterpret_cast<PyObject*>(ty) != var_ty)
        return raise(PyExc_TypeError, "Expected an ir.Var, not ", ty->tp_name);

    // Loosely typed scalars become Python constants
    PyPtr loose_ty = steal(PyObject_CallMethod(var.get(), "get_loose_type_allow_invalid", nullptr));
    if (!loose_ty) return ErrorRaised;
    static LazyImporter loosely_typed_scalar_importer;
    Result<bool> is_loose = is_ir_type(loose_ty.get(), &loosely_typed_scalar_importer,
                                       "LooselyTypedScalar");
    if (!is_loose.is_ok()) return ErrorRaised;
    if (*is_loose) {
        PyPtr constant = getattr(loose_ty, "value");
        if (!constant) return ErrorRaised;
        return borrowed_value_from_pyobject(constant.get()).to_owned();
    }

    PyPtr ir_ty = steal(PyObject_CallMethod(var.get(), "get_type_allow_invalid", nullptr));
    if (!ir_ty) return ErrorRaised;

    static LazyImporter tensor_like_ty_importer;
    Result<bool> is_tensor_like = is_ir_type(ir_ty.get(), &tensor_like_ty_importer,
                                             "TensorLikeTy");
    if (!is_tensor_like.is_ok()) return ErrorRaised;
    if (*is_tensor_like) {
        PyPtr sym = steal(PyObject_CallMethod(ir_ty.get(), "make_symbol", "(O)", var.get()));
        if (!sym) return ErrorRaised;
        if (!is_tensor_pytype(Py_TYPE(sym.get())))
            return raise(PyExc_TypeError,
                         "Expected make_symbol() to return a tensor-like object, not ",
                         Py_TYPE(sym.get())->tp_name);
        return BorrowedValue::from_tensor(reinterpret_cast<TensorBase*>(sym.release())).steal();
    }

    static LazyImporter none_type_importer;
    Result<bool> is_none = is_ir_type(ir_ty.get(), &none_type_importer, "NoneType");
    if (!is_none.is_ok()) return ErrorRaised;
    if (*is_none)
        return Value::none();

    static LazyImporter tuple_ty_importer;
    Result<bool> is_tuple = is_ir_type(ir_ty.get(), &tuple_ty_importer, "TupleTy");
    if (!is_tuple.is_ok()) return ErrorRaised;
    if (*is_tuple) {
        PyPtr item_types = steal(PyObject_CallMethod(ir_ty.get(), "aggregate_item_types",
                                                     nullptr));
        if (!item_types) return ErrorRaised;
        Py_ssize_t size = PyObject_Length(item_types.get());
        if (size < 0) return ErrorRaised;
        RefPtr<TupleValue> tuple = TupleValue::create_undefined(size);
        if (!aggregate_items_from_ir_var(var.get(), tuple->items(), size)) return ErrorRaised;
        return Value::from_tuple_value(std::move(tuple));
    }

    static LazyImporter slice_type_importer;
    Result<bool> is_slice = is_ir_type(ir_ty.get(), &slice_type_importer, "SliceType");
    if (!is_slice.is_ok()) return ErrorRaised;
    PyPtr is_slice_aggregate;
    if (*is_slice) {
        is_slice_aggregate = steal(PyObject_CallMethod(ir_ty.get(), "is_aggregate", nullptr));
        if (!is_slice_aggregate) return ErrorRaised;
    }
    if (*is_slice && is_slice_aggregate.get() == Py_True) {
        Value items[3];
        if (!aggregate_items_from_ir_var(var.get(), items, 3)) return ErrorRaised;
        RefPtr<SliceValue> slice = ::steal(new SliceValue());
        slice->start = std::move(items[0]);
        slice->stop = std::move(items[1]);
        slice->step = std::move(items[2]);
        return Value::from_slice_value(std::move(slice));
    }

    static LazyImporter dict_ty_importer;
    Result<bool> is_dict = is_ir_type(ir_ty.get(), &dict_ty_importer, "DictTy");
    if (!is_dict.is_ok()) return ErrorRaised;
    if (*is_dict) {
        PyPtr keys = getattr(ir_ty, "keys");
        if (!keys) return ErrorRaised;
        CHECK(PyTuple_Check(keys.get()));
        size_t size = PyTuple_GET_SIZE(keys.get());
        RefPtr<DictStringKeys> dict_keys = DictStringKeys::create(size);
        for (size_t i = 0; i < size; ++i) {
            PyObject* key = Py_NewRef(PyTuple_GET_ITEM(keys.get(), i));
            CHECK(PyUnicode_CheckExact(key));
            PyUnicode_InternInPlace(&key);
            (*dict_keys)[i] = steal(key);
        }
        RefPtr<DictValue> dict = DictValue::create_undefined(std::move(dict_keys));
        if (!aggregate_items_from_ir_var(var.get(), dict->values(), size)) return ErrorRaised;
        return Value::from_dict_value(std::move(dict));
    }

    // Other constants that Value supports, e.g. strings and enums
    PyPtr is_constant = steal(PyObject_CallMethod(var.get(), "is_constant", nullptr));
    if (!is_constant) return ErrorRaised;
    if (is_constant.get() == Py_True) {
        PyPtr constant = steal(PyObject_CallMethod(var.get(), "get_constant", nullptr));
        if (!constant) return ErrorRaised;
        BorrowedValue value = borrowed_value_from_pyobject(constant.get());
        if (!value.is_unsupported_pyobject())
            return value.to_owned();
    }

    return BorrowedValue::from_misc_ir_var(var.release()).steal();
}


static PyPtr value_to_ir_var(BorrowedValue value);

// Converts the values returned by get_item(i) to ir.Var objects
// and passes them as a tuple to the given core_ops function
template <typename GetItem>
static PyPtr call_with_ir_var_tuple(LazyImporter* importer, const char* func_name,
                                    size_t num_values, GetItem&& get_item) {
    PyPtr vars = steal(PyTuple_New(num_values));
    if (!vars) return {};
    for (size_t i = 0; i < num_values; ++i) {
        PyPtr var = value_to_ir_var(get_item(i));
        if (!var) return {};
        PyTuple_SET_ITEM(vars.get(), i, var.release());
    }

    PyObject* func = importer->get("cuda.tile._ir.core_ops", func_name);
    if (!func) return {};
    return steal(PyObject_CallOneArg(func, vars.get()));
}

static PyPtr value_to_ir_var(BorrowedValue value) {
    if (value.is_tensor())
        return tensor_to_ir_var(value.as_tensor());
    if (value.is_misc_ir_var())
        return newref(value.as_misc_ir_var());

    if (value.is_undefined() || value.is_hir_value_id()) {
        raise(PyExc_AssertionError, "Can't convert value '", value, "' to ir.Var");
        return {};
    }

    // Tuples, slices and dicts may be nested arbitrarily deep
    RecursionGuard guard(" while converting a Value to an ir.Var");
    if (!guard) return {};

    if (value.is_tuple()) {
        static LazyImporter build_tuple_importer;
        return call_with_ir_var_tuple(&build_tuple_importer, "build_tuple", value.tuple_size(),
                                      [&] (size_t i) { return value.tuple_item(i); });
    }

    if (value.is_slice()) {
        RefPtr<SliceValue> slice = value.as_slice();
        const Value* items[3] = {&slice->start, &slice->stop, &slice->step};
        static LazyImporter build_slice_importer;
        return call_with_ir_var_tuple(&build_slice_importer, "build_slice", 3,
                                      [&] (size_t i) { return items[i]->borrow(); });
    }

    if (value.is_dict()) {
        const DictValue& dict = value.as_borrowed_dict_value();
        PyPtr keys = steal(PyTuple_New(dict.size()));
        if (!keys) return {};
        PyPtr values = steal(PyTuple_New(dict.size()));
        if (!values) return {};
        for (size_t i = 0; i < dict.size(); ++i) {
            PyTuple_SET_ITEM(keys.get(), i, Py_NewRef(dict.key(i)));
            PyPtr var = value_to_ir_var(dict.value(i).borrow());
            if (!var) return {};
            PyTuple_SET_ITEM(values.get(), i, var.release());
        }
        static LazyImporter build_dict_importer;
        PyObject* build_dict = build_dict_importer.get("cuda.tile._ir.core_ops", "build_dict");
        if (!build_dict) return {};
        return steal(PyObject_CallFunctionObjArgs(build_dict, keys.get(), values.get(), nullptr));
    }

    // Other constants, e.g. None, numbers, strings and enums
    PyPtr obj = value.to_pyobject();
    if (!obj) return {};
    static LazyImporter sym2var_importer;
    PyObject* sym2var = sym2var_importer.get("cuda.tile._ir.core_ops", "sym2var");
    if (!sym2var) return {};
    return steal(PyObject_CallFunctionObjArgs(sym2var, obj.get(), Py_True, nullptr));
}


template <>
struct HirCoroutineImpl<HirCoroutineKind::PythonCoroutine> {
    PyPtr coro;

    explicit HirCoroutineImpl(PyPtr coro) : coro(std::move(coro)) {}

    ~HirCoroutineImpl() {
        if (coro) {
            // Best effort clean up on internal error.
            // (On normal exit, coro would be already cleared).
            ErrorGuard guard;
            Py_XDECREF(PyObject_CallMethod(coro.get(), "close", ""));
            if (PyErr_Occurred())
                PyErr_Print();
        }
    }

    Result<bool> run(HirRunner* runner, CoroutineValue* cv) {
        CHECK(coro);
        if (!cv->exc) {
            // Happy path: use PyIter_Send() C API for efficiency
            PyObject* res = nullptr;
            PySendResult send_res = PyIter_Send(coro.get(), cv->value.get(), &res);
            switch (send_res) {
            case PYGEN_RETURN:
                cv->set_success(steal(res));
                coro.reset();
                return true;
            case PYGEN_NEXT:
                cv->set_success(newref(Py_None));
                if (!runner->push_continuation(steal(res), false))
                    return ErrorRaised;
                return false;
            case PYGEN_ERROR:
                cv->save_raised_exception();
                coro.reset();
                return true;
            }
            CHECK_UNREACHABLE;
        }

        // Slow path: need to call .throw() since there is no public C API to _gen_throw()
        PyPtr continuation = steal(PyObject_CallMethod(
                    coro.get(), "throw", "(O)", cv->exc.value.get()));
        if (continuation) {
            cv->set_success(newref(Py_None));
            if (!runner->push_continuation(std::move(continuation), false))
                return ErrorRaised;
            return false;
        } else {
            CHECK(PyErr_Occurred());
            cv->save_raised_exception();
            if (PyErr_GivenExceptionMatches(cv->exc.value.get(), PyExc_StopIteration)) {
                PyPtr ret = getattr(cv->exc.value, "value");
                if (!ret) return ErrorRaised;
                cv->set_success(std::move(ret));
            }
            coro.reset();
            return true;
        }
    }
};


enum class [[nodiscard]] HirEvalStatus {
    OK,
    DelegatedToCoroutine,
    Error,
    Suspended,
    GeneratorYielded,
    FunctionFinished,
};


template <>
struct HirEvalContext<false> {
    HirRunner* runner;
    HirFrame* frame;

    template <HirOpcode Opcode, typename... Args>
    HirEvalStatus delegate_to(Value* result, Args&&... args) {
        return OpcodeImpl<Opcode>::run(this, result, std::forward<Args>(args)...);
    }

    HirEvalStatus delegate_to_python_coroutine(PyPtr coro) {
        runner->push_coroutine<HirCoroutineKind::PythonCoroutine>(std::move(coro));
        return HirEvalStatus::DelegatedToCoroutine;
    }

    // Raises a compile-time error
    template <typename... Args>
    HirEvalStatus error(Error::Tag tag, Args&&... message) {
        make_error(tag, std::forward<Args>(message)...).raise_compile_time_error();
        return HirEvalStatus::Error;
    }
};

struct HirBuilder;

template <>
struct HirEvalContext<true> {
    HirRunner* runner;
    PyObject* cur_loc;
    HirBuilder* builder;

    // Defined after HirBuilder
    template <HirOpcode Opcode, typename... Args>
    HirEvalStatus delegate_to(Value* result, Args&&... args);

    HirEvalStatus suspend() {
        return HirEvalStatus::Suspended;
    }

    // Delays the error until full evaluation
    template <typename... Args>
    HirEvalStatus error(Error::Tag, Args&&...) {
        return suspend();
    }
};

static inline BorrowedValue resolve_operand(HirWord word, Value* value_map) {
    BorrowedValue val = BorrowedValue::from_raw_data(word.u64);
    if (val.is_hir_value_id())
        return value_map[val.as_hir_value_id()].borrow();
    else
        return val;
}

template <uint32_t... Idx>
static auto resolve_operands_fixed(std::integer_sequence<uint32_t, Idx...>,
                                   const HirWord* operand_words, Value* value_map) {
    return std::make_tuple(resolve_operand(operand_words[Idx], value_map)...);
}

template <typename NumOperands>
static auto resolve_operands(const HirWord* operand_words, NumOperands num_operands,
                             HirEvalContext<false>* ctx, Value* value_map) {
    if constexpr (std::is_same_v<NumOperands, uint32_t>) {
        // Variadic operands
        BorrowedValue* resolved_operands = ctx->runner->stack.allocate<BorrowedValue>(
                num_operands);
        static_assert(std::is_trivially_destructible_v<BorrowedValue>);
        for (uint32_t i = 0; i < num_operands; ++i) {
            BorrowedValue val = resolve_operand(operand_words[i], value_map);
            new (&resolved_operands[i]) BorrowedValue(val);
        }
        return std::make_tuple(resolved_operands, num_operands);
    } else {
        // Fixed operands
        static_assert(is_u32_constant_v<NumOperands>);
        return resolve_operands_fixed(
                std::make_integer_sequence<uint32_t, NumOperands::value>(),
                operand_words,
                value_map);
    }
}

template <HirOpcode Opcode>
static HirEvalStatus run_opcode(HirEvalContext<false>* ctx, HirWord** cursor, Value* value_map,
                                uint32_t instruction_index) {
    auto [operand_words, num_operands, header_meta_tuple, attrs_tuple]
            = parse_instruction<Opcode>(cursor);
    auto operand_args = resolve_operands(operand_words, num_operands, ctx, value_map);
    return std::apply(
        [&] (auto&&... unpacked) {
            if constexpr (kOpcodeRequiresLoc<Opcode>) {
                HirProgram& prog = py_unwrap<HirProgram>(ctx->frame->program);
                return ctx->delegate_to<Opcode>(
                        &value_map[instruction_index], prog.locations[instruction_index].get(),
                        unpacked...);
            } else {
                return ctx->delegate_to<Opcode>(&value_map[instruction_index], unpacked...);
            }
        }, std::tuple_cat(operand_args, header_meta_tuple, attrs_tuple));
}


static PyPtr eval_block_awaitable_new(HirFrame* frame, InstructionPointer iptr,
                                      Value block_parameter) {
    EvalBlockAwaitable awaitable = {newref(frame), iptr, std::move(block_parameter)};
    return py_create_object_with_args<EvalBlockAwaitable>(std::move(awaitable));
}


// The result of evaluating a block that executed GeneratorYield
struct HirGeneratorYielded {
    PyPtr value;  // the yielded ir.Var
    RefPtr<HirFrame> frame;
    InstructionPointer resume_iptr;  // right after the GeneratorYield instruction

    static PyTypeObject pytype;
};

static PyObject* HirGeneratorYielded_get_value(PyObject* self, void*) {
    return Py_NewRef(py_unwrap<HirGeneratorYielded>(self).value.get());
}

static PyGetSetDef HirGeneratorYielded_getset[] = {
    {"value", HirGeneratorYielded_get_value, nullptr, nullptr, nullptr},
    {}
};

// Returns a new awaitable that evaluates the rest of the block after the GeneratorYield.
// Can be called multiple times, e.g. to exit a context manager on multiple paths.
static PyObject* HirGeneratorYielded_resume(PyObject* self, PyObject*) {
    HirGeneratorYielded& yielded = py_unwrap<HirGeneratorYielded>(self);
    return eval_block_awaitable_new(yielded.frame.get(), yielded.resume_iptr, Value()).release();
}

static PyMethodDef HirGeneratorYielded_methods[] = {
    {"resume", HirGeneratorYielded_resume, METH_NOARGS, nullptr},
    {}
};

PyTypeObject HirGeneratorYielded::pytype = {
    .tp_name = "cuda.tile._cext.HirGeneratorYielded",
    .tp_basicsize = sizeof(PythonWrapper<HirGeneratorYielded>),
    .tp_dealloc = pywrapper_dealloc<HirGeneratorYielded>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = HirGeneratorYielded_methods,
    .tp_getset = HirGeneratorYielded_getset,
};

// The type of HIR_BLOCK_TERMINATED, a singleton returned by the Python implementations
// of the jump opcodes (Continue, Break, Return). When a block receives it from an opcode,
// the block stops and returns it to its awaiter. This way, a jump inside a flattened branch
// (e.g. `if True: break`) also terminates the enclosing blocks, up to the jump's target.
static PyObject* HirBlockTerminated_repr(PyObject*) {
    return PyUnicode_FromString("HIR_BLOCK_TERMINATED");
}

static PyTypeObject HirBlockTerminated_pytype = {
    .tp_name = "cuda.tile._cext.HirBlockTerminated",
    .tp_basicsize = sizeof(PyObject),
    .tp_repr = HirBlockTerminated_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

static PyObject* g_hir_block_terminated;

static PyPtr hir_generator_yielded_new(const Value& value, HirFrame* frame,
                                       InstructionPointer resume_iptr) {
    PyPtr var = value_to_ir_var(value.borrow());
    if (!var) return {};
    return py_create_object_with_args<HirGeneratorYielded>(
            HirGeneratorYielded{std::move(var), newref(frame), resume_iptr});
}

static Status do_set_error_loc(SavedException* exc, PyObject* loc, const HirFrame& frame) {
    static LazyImporter tile_error_importer;
    PyObject* tile_error = tile_error_importer.get("cuda.tile._exception", "TileError");
    if (!tile_error) return ErrorRaised;
    CHECK(PyType_Check(tile_error));
    if (!PyObject_TypeCheck(exc->value.get(), reinterpret_cast<PyTypeObject*>(tile_error))) {
        if (!PyObject_TypeCheck(exc->value.get(), reinterpret_cast<PyTypeObject*>(PyExc_Exception)))
            return OK;

        // Wrap other exceptions in an InternalError
        make_error(Error::kInternalError, exc->value.get()).raise_compile_time_error();
        SavedException wrapped = save_raised_exception();
        wrapped.normalize();
        PyException_SetCause(wrapped.value.get(), exc->value.release());
        *exc = std::move(wrapped);
    }

    PyObject* value = exc->value.get();
    PyPtr old_loc = getattr(value, "loc");
    if (!old_loc) return ErrorRaised;
    PyPtr is_unknown = steal(PyObject_CallMethod(old_loc.get(), "is_unknown", nullptr));
    if (!is_unknown) return ErrorRaised;
    if (is_unknown.get() != Py_True) return OK;

    PyObject* call_site = frame.call_site_loc ? frame.call_site_loc.get() : Py_None;
    PyPtr new_loc = steal(PyObject_CallMethod(
            loc, "retarget", "OO", call_site, frame.function_desc.get()));
    if (!new_loc) return ErrorRaised;
    if (PyObject_SetAttrString(value, "loc", new_loc.get()) < 0) return ErrorRaised;
    return OK;
}

// Replaces the saved exception with the currently raised one, chained to the original
static void chain_raised_exception(SavedException* exc) {
    SavedException original = std::move(*exc);
    *exc = save_raised_exception();
    exc->normalize();
    PyException_SetCause(exc->value.get(), original.value.release());
}

static void set_error_loc(SavedException* exc, PyObject* loc, const HirFrame& frame) {
    if (!do_set_error_loc(exc, loc, frame))
        chain_raised_exception(exc);
}


template <>
struct HirCoroutineImpl<HirCoroutineKind::EvalBlock> {
    RefPtr<HirFrame> frame;
    InstructionPointer block_start;
    InstructionPointer program_counter;
    bool resuming = false;
    bool dump_hir_on_error;
    GrowableStack::Watermark stack_base;

    explicit HirCoroutineImpl(RefPtr<HirFrame> frame, InstructionPointer block_start,
                              bool dump_hir_on_error)
        : frame(std::move(frame)), block_start(block_start), program_counter(block_start),
          dump_hir_on_error(dump_hir_on_error) {}

    // Prints the block to stderr, marking the given instruction
    Status dump_hir(HirProgram& prog, uint32_t marked_instruction_id) {
        StringBuilder sb;
        sb.append_many("==== HIR for ^", block_start.index, " ====\n");
        print_block(prog, block_start, 0, marked_instruction_id, &sb);
        PyPtr str = sb.build();
        if (!str) return ErrorRaised;
        PySys_FormatStderr("%U\n", str.get());
        return OK;
    }

    void handle_error(uint32_t instruction_index, CoroutineValue* cv) {
        HirProgram& prog = py_unwrap<HirProgram>(frame->program);
        set_error_loc(&cv->exc, prog.locations[instruction_index].get(), *frame);
        if (dump_hir_on_error && !dump_hir(prog, instruction_index))
            chain_raised_exception(&cv->exc);
    }

    static void set_success_or_exception(PyPtr new_value, CoroutineValue* cv) {
        if (new_value)
            cv->set_success(std::move(new_value));
        else
            cv->save_raised_exception();
    }

    Result<bool> run(HirRunner* runner, CoroutineValue* cv) {
        Result<bool> res = do_run(runner, cv);
        // On error, program_counter points at the instruction that has failed
        if (cv->exc) handle_error(program_counter.index, cv);
        return res;
    }

    // On error, saves the exception in `cv` and returns true
    Result<bool> do_run(HirRunner* runner, CoroutineValue* cv) {
        HirProgram& prog = py_unwrap<HirProgram>(frame->program);
        const HirWord* end = prog.data.data() + prog.data.size();
        HirWord* instruction_data = prog.data.data() + program_counter.offset;
        size_t instruction_index = program_counter.index;
        Value* value_map = frame->value_map();

        if (resuming) {
            // The coroutine that we delegated to has completed.
            if (cv->exc) return true;  // Propagate the exception
            if (cv->value.get() == g_hir_block_terminated)
                return true;  // A jump has terminated the block. Pass the sentinel to the awaiter.
            Result<Value> r = value_from_ir_var(std::move(cv->value));
            if (!r.is_ok()) {
                cv->save_raised_exception();
                return true;
            }
            cv->set_success(newref(Py_None));
            value_map[instruction_index++] = std::move(*r);
        } else {
            stack_base = runner->stack.current_watermark();
        }

        HirEvalContext<false> ctx = {runner, frame.get()};

        // The main interpreter loop
        while (instruction_data < end) {
            // Free the temporary data of the previous instruction, including the one
            // that has delegated to a coroutine if we are resuming
            runner->stack.reset_to(stack_base);
            HirOpcode opcode = header_opcode(*instruction_data);
            HirEvalStatus status = HirEvalStatus::OK;
            switch (opcode) {
                #define DISPATCH_OPCODE(name) \
                    case HirOpcode::name: \
                        status = run_opcode<HirOpcode::name>( \
                                &ctx, &instruction_data, value_map, instruction_index); \
                        if (status == HirEvalStatus::OK) { \
                            ++instruction_index; \
                            continue; \
                        } \
                        break;

                FOREACH_HIR_OPCODE(DISPATCH_OPCODE)
                #undef DISPATCH_OPCODE

            case HirOpcode::BeginBlock:
                // Skip the nested block
                {
                    InstructionPointer iptr = instruction_data[1].iptr;
                    instruction_data = prog.data.data() + iptr.offset;
                    instruction_index = iptr.index;
                    continue;
                }
            case HirOpcode::EndBlock:
                goto out;
            }

            // Handle non-OK status.
            // NOTE: The offset is pointing at the next instruction's data already, because
            // run_opcode() has advanced it; but we don't advance the index: on error, it identifies
            // the failed instruction, and on resume, we need to store the result Value into
            // the appropriate value_map slot.
            program_counter.offset = instruction_data - prog.data.data();
            program_counter.index = instruction_index;
            switch (status) {
            case HirEvalStatus::OK:
                CHECK_UNREACHABLE;  // checked by every opcode already
            case HirEvalStatus::DelegatedToCoroutine:
                // Opcode implementation has delegated to a coroutine.
                // Suspend the current coroutine until the new one finishes.
                resuming = true;
                cv->set_success(newref(Py_None));
                return false;
            case HirEvalStatus::Error:
                cv->save_raised_exception();
                return true;
            case HirEvalStatus::Suspended:
                CHECK_UNREACHABLE;  // should have been converted to OK by delegate_to()
            case HirEvalStatus::GeneratorYielded:
                {
                    InstructionPointer resume_iptr = {
                        static_cast<uint32_t>(instruction_data - prog.data.data()),
                        static_cast<uint32_t>(instruction_index + 1),
                    };
                    PyPtr ret = hir_generator_yielded_new(
                            value_map[instruction_index], frame.get(), resume_iptr);
                    set_success_or_exception(std::move(ret), cv);
                    // Like in Python, the yield expression evaluates to None
                    value_map[instruction_index] = Value::none();
                }
                return true;
            case HirEvalStatus::FunctionFinished:
                {
                    // Return the function's result to the awaiter
                    PyPtr ret = value_to_ir_var(value_map[instruction_index].borrow());
                    set_success_or_exception(std::move(ret), cv);
                }
                return true;
            }
        }
    out:
        // We're done. Blocks other than function bodies don't have a result.
        cv->set_success(newref(Py_None));
        return true;
    }
};


static void set_traceback_from_python_coro(HirRunner* runner) {
    CHECK(PyErr_Occurred());
    for (HirCoroutine* c = runner->top; c; c = c->prev) {
        if (c->kind == HirCoroutineKind::PythonCoroutine) {
            using Impl = HirCoroutineImpl<HirCoroutineKind::PythonCoroutine>;
            Impl* impl = static_cast<Impl*>(c->impl);
            if (impl->coro) {
                PyPtr frame = try_getattr(impl->coro, "cr_frame");
                if (frame && PyFrame_Check(frame.get()))
                    PyTraceBack_Here(reinterpret_cast<PyFrameObject*>(frame.get()));
            }
            break;
        }
    }
}

static PyPtr hir_run(HirRunner* runner) {
    CoroutineValue cv = {newref(Py_None), {}};
    while (true) {
        Result<bool> res = runner->top->dispatch(
            [&] (auto impl) {
                Result<bool> r = impl->run(runner, &cv);
                if (r.is_ok() && *r) std::destroy_at(impl);
                return r;
            });

        if (!res.is_ok()) {
            // Unwind on internal error
            set_traceback_from_python_coro(runner);
            runner->clear();
            return {};
        }

        if (*res) {
            runner->pop();
            if (runner->empty()) break;
        }
    }

    if (cv.exc) {
        cv.exc.restore();
        return {};
    } else {
        return std::move(cv.value);
    }
}


static PyObject* py_hir_run(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1 && nargs != 2) {
        raise(PyExc_TypeError,
              "hir_run(awaitable, [dump_hir_on_error,]) expected 1 or 2 arguments");
        return nullptr;
    }
    PyObject* dump_hir_on_error = nargs == 2 ? args[1] : Py_False;
    if (!PyBool_Check(dump_hir_on_error)) {
        raise(PyExc_TypeError, "dump_hir_on_error must be a bool, not ",
              Py_TYPE(dump_hir_on_error)->tp_name);
        return nullptr;
    }

    HirRunner runner;
    runner.dump_hir_on_error = dump_hir_on_error == Py_True;
    if (!runner.push_continuation(newref(args[0]), true)) return nullptr;

    return hir_run(&runner).release();
}

static Result<std::pair<const BorrowedValue*, size_t>> unpack_starred_args(
        Accessor<BitmapAttr> starred_bitmap, const BorrowedValue* args, GrowableStack* stack) {
    size_t num_args = starred_bitmap.attr.num_bits;
    if (starred_bitmap.is_zero())
        return std::make_pair(args, num_args);

    // First pass: compute the number of unpacked items
    size_t num_unpacked = 0;
    for (size_t i = 0; i < num_args; ++i) {
        if (starred_bitmap[i]) {
            if (!args[i].is_tuple()) {
                return make_error(Error::kTypeError, "Expected a tuple after *, not ",
                                  use_type(args[i])).raise_compile_time_error();
            }
            num_unpacked += args[i].tuple_size();
        } else {
            ++num_unpacked;
        }
    }

    // Second pass: unpack
    BorrowedValue* unpacked = stack->allocate<BorrowedValue>(num_unpacked);
    size_t pos = 0;
    for (size_t i = 0; i < num_args; ++i) {
        if (starred_bitmap[i]) {
            size_t size = args[i].tuple_size();
            for (size_t j = 0; j < size; ++j)
                new (&unpacked[pos++]) BorrowedValue(args[i].tuple_item(j));
        } else {
            new (&unpacked[pos++]) BorrowedValue(args[i]);
        }
    }
    CHECK(pos == num_unpacked);
    return std::make_pair(static_cast<const BorrowedValue*>(unpacked), num_unpacked);
}

template <>
struct OpcodeImpl<HirOpcode::GenericCall> {
    constexpr static size_t kArgCountBits = HirWord::kHeaderDataBits / 2;
    constexpr static size_t kArgCountMask = (1u << kArgCountBits) - 1;

    static auto make_layout(uint32_t num_pos, uint32_t num_kw) {
        return make_opcode_layout<true>(
            1 + num_pos + num_kw,
            {},
            std::make_tuple(BitmapAttr{num_pos}, StringSeqAttr{num_kw})
        );
    }

    static auto parse_layout(HirWord header) {
        return make_layout(header.header.data & kArgCountMask,
                           header.header.data >> kArgCountBits);
    }

    static uint64_t encode_header_data(size_t num_operands,
                                       Accessor<BitmapAttr> starred_bitmap,
                                       Accessor<StringSeqAttr> keywords) {
        size_t num_pos = starred_bitmap.attr.num_bits;
        size_t num_kw = keywords.attr.num_strings;
        CHECK(num_operands == 1 + num_pos + num_kw);
        CHECK(num_pos <= kArgCountMask);
        CHECK(num_kw <= kArgCountMask);
        return num_pos | (num_kw << kArgCountBits);
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             const BorrowedValue* operands,
                             size_t num_operands,
                             Accessor<BitmapAttr> starred_bitmap,
                             Accessor<StringSeqAttr> keywords) {
        if constexpr (PartialEval) {
            return ctx->suspend();  // TODO: partial evaluation
        } else {
            size_t num_kwargs = keywords.attr.num_strings;
            CHECK(num_operands >= num_kwargs + 1);
            size_t num_pos_args = num_operands - num_kwargs - 1;

            PyPtr callee_var = value_to_ir_var(operands[0]);
            if (!callee_var) return HirEvalStatus::Error;

            // Unpack positional arguments and convert them to a tuple of IR variables
            CHECK(starred_bitmap.attr.num_bits == num_pos_args);
            Result<std::pair<const BorrowedValue*, size_t>> unpacked = unpack_starred_args(
                    starred_bitmap, operands + 1, &ctx->runner->stack);
            if (!unpacked.is_ok()) return HirEvalStatus::Error;
            auto [pos_args, num_unpacked] = *unpacked;

            PyPtr args_tuple = steal(PyTuple_New(num_unpacked));
            if (!args_tuple) return HirEvalStatus::Error;
            for (size_t i = 0; i < num_unpacked; ++i) {
                PyPtr var = value_to_ir_var(pos_args[i]);
                if (!var) return HirEvalStatus::Error;
                PyTuple_SET_ITEM(args_tuple.get(), i, var.release());
            }

            // Make a dict of keyword arguments
            PyPtr kwargs_dict = steal(PyDict_New());
            if (!kwargs_dict) return HirEvalStatus::Error;
            const BorrowedValue* kwarg_values = operands + 1 + num_pos_args;
            auto add_kwarg = [&] (PyObject* keyword, BorrowedValue val) -> HirEvalStatus {
                int contains = PyDict_Contains(kwargs_dict.get(), keyword);
                if (contains < 0) return HirEvalStatus::Error;
                if (contains) {
                    return ctx->error(Error::kTypeError,
                                      "Got multiple values for keyword argument '", keyword, "'");
                }
                PyPtr var = value_to_ir_var(val);
                if (!var) return HirEvalStatus::Error;
                if (PyDict_SetItem(kwargs_dict.get(), keyword, var.get()) < 0)
                    return HirEvalStatus::Error;
                return HirEvalStatus::OK;
            };
            for (size_t i = 0; i < num_kwargs; ++i) {
                PyObject* keyword = keywords[i];
                BorrowedValue val = kwarg_values[i];
                if (keyword) {
                    HirEvalStatus status = add_kwarg(keyword, val);
                    if (status != HirEvalStatus::OK) return status;
                } else {
                    if (!val.is_dict())
                        return ctx->error(Error::kTypeError, "Expected a dict after **, not ",
                                          use_type(val));
                    const DictValue& dict = val.as_borrowed_dict_value();
                    for (size_t j = 0; j < dict.size(); ++j) {
                        HirEvalStatus status = add_kwarg(dict.key(j), dict.value(j).borrow());
                        if (status != HirEvalStatus::OK) return status;
                    }
                }
            }

            // Call hir2ir.generic_call() to get a Python coroutine
            static LazyImporter generic_call_importer;
            PyObject* generic_call = generic_call_importer.get("cuda.tile._passes.hir2ir",
                                                               "generic_call");
            if (!generic_call) return HirEvalStatus::Error;
            PyPtr coro = steal(PyObject_CallFunctionObjArgs(
                        generic_call, callee_var.get(), args_tuple.get(), kwargs_dict.get(), loc,
                        nullptr));
            if (!coro) return HirEvalStatus::Error;
            return ctx->delegate_to_python_coroutine(std::move(coro));
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::LoadLocal> {
    static constexpr size_t kDepthBits = HirWord::kHeaderDataBits - 32;

    static auto make_layout(uint32_t depth, uint32_t index) {
        return make_opcode_layout<true>(
                std::integral_constant<uint32_t, 0>{}, std::make_tuple(depth, index), {});
    }

    static auto parse_layout(HirWord header) {
        uint64_t data = header.header.data;
        uint32_t depth = data >> 32;
        uint32_t index = data & 0xffff'ffff;
        return make_layout(depth, index);
    }

    static uint64_t encode_header_data(uint32_t depth, uint32_t index) {
        CHECK(depth < (1 << kDepthBits));
        return (static_cast<uint64_t>(depth) << 32) | index;
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             uint32_t depth,
                             uint32_t index) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            static LazyImporter load_local_importer;
            PyObject* load_local = load_local_importer.get("cuda.tile._ir.core_ops", "load_local");
            if (!load_local) return HirEvalStatus::Error;
            PyPtr var = steal(PyObject_CallFunction(load_local, "IIO", depth, index, loc));
            if (!var) return HirEvalStatus::Error;
            Result<Value> r = value_from_ir_var(std::move(var));
            if (!r.is_ok()) return HirEvalStatus::Error;
            *result = std::move(*r);
            return HirEvalStatus::OK;
        }
    }
};


template <>
struct OpcodeImpl<HirOpcode::StoreLocal> {
    static auto make_layout(uint32_t index) {
        return make_opcode_layout<true>(
                std::integral_constant<uint32_t, 1>{}, std::make_tuple(index), {});
    }

    static auto parse_layout(HirWord header) {
        return make_layout(header.header.data);
    }

    static uint64_t encode_header_data(uint32_t index) {
        return index;
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             BorrowedValue value,
                             uint32_t index) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            PyPtr var = value_to_ir_var(value);
            if (!var) return HirEvalStatus::Error;

            static LazyImporter store_local_importer;
            PyObject* store_local = store_local_importer.get("cuda.tile._ir.core_ops",
                                                             "store_local");
            if (!store_local) return HirEvalStatus::Error;
            PyPtr res = steal(PyObject_CallFunction(store_local, "IOO", index, var.get(), loc));
            if (!res) return HirEvalStatus::Error;
            *result = Value::none();
            return HirEvalStatus::OK;
        }
    }
};


template <>
struct OpcodeImpl<HirOpcode::LoadGlobal> {
    static auto make_layout() {
        return make_opcode_layout<true>(std::integral_constant<uint32_t, 1>{}, {}, {});
    }

    static auto parse_layout(HirWord) {
        return make_layout();
    }

    static uint64_t encode_header_data() { return 0; }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             BorrowedValue value) {
        if (!value.is_unsupported_pyobject()) {
            *result = value.to_owned();
            return HirEvalStatus::OK;
        }

        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            static LazyImporter load_global_importer;
            PyObject* load_global = load_global_importer.get("cuda.tile._ir.core_ops",
                                                             "load_global");
            if (!load_global) return HirEvalStatus::Error;
            PyPtr obj = value.to_pyobject();
            if (!obj) return HirEvalStatus::Error;
            PyPtr var = steal(PyObject_CallFunctionObjArgs(load_global, obj.get(), loc, nullptr));
            if (!var) return HirEvalStatus::Error;
            Result<Value> r = value_from_ir_var(std::move(var));
            if (!r.is_ok()) return HirEvalStatus::Error;
            *result = std::move(*r);
            return HirEvalStatus::OK;
        }
    }
};


template <>
struct OpcodeImpl<HirOpcode::LoadUndefinedVariable> {
    static auto make_layout() {
        return make_opcode_layout(
            std::integral_constant<uint32_t, 0>{},
            {},
            std::make_tuple(PyobjectAttr{})  // name of the variable (PyUnicode)
        );
    }

    static auto parse_layout(HirWord) {
        return make_layout();
    }

    static uint64_t encode_header_data(Accessor<PyobjectAttr>) { return 0; }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             Accessor<PyobjectAttr> name) {
        return ctx->error(Error::kSyntaxError, "Undefined variable ", name.get(), " used");
    }
};

template <>
struct OpcodeImpl<HirOpcode::BuildFormattedString> {
    static auto make_layout(uint32_t num_values) {
        return make_opcode_layout<true>(
            num_values,
            {},
            std::make_tuple(
                PyobjectAttr{},  // StringFormat
                PyobjectAttr{}  // tuple of the values' locations
            )
        );
    }

    static auto parse_layout(HirWord header) {
        return make_layout(header.header.data);
    }

    static uint64_t encode_header_data(size_t num_operands,
                                       Accessor<PyobjectAttr>,
                                       Accessor<PyobjectAttr>) {
        return num_operands;
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             const BorrowedValue* operands,
                             size_t num_operands,
                             Accessor<PyobjectAttr> format,
                             Accessor<PyobjectAttr> value_locs) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            PyPtr values = steal(PyTuple_New(num_operands));
            if (!values) return HirEvalStatus::Error;
            for (size_t i = 0; i < num_operands; ++i) {
                PyPtr var = value_to_ir_var(operands[i]);
                if (!var) return HirEvalStatus::Error;
                PyTuple_SET_ITEM(values.get(), i, var.release());
            }

            static LazyImporter build_formatted_string_importer;
            PyObject* build_formatted_string = build_formatted_string_importer.get(
                    "cuda.tile._ir.core_ops", "build_formatted_string");
            if (!build_formatted_string) return HirEvalStatus::Error;
            PyPtr coro = steal(PyObject_CallFunctionObjArgs(
                        build_formatted_string, format.get(), values.get(), value_locs.get(), loc,
                        nullptr));
            if (!coro) return HirEvalStatus::Error;
            return ctx->delegate_to_python_coroutine(std::move(coro));
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::BuildTuple> {
    static auto make_layout(uint32_t num_items) {
        return make_opcode_layout(
            num_items,
            {},
            std::make_tuple(BitmapAttr{num_items})  // starred_bitmap
        );
    }

    static auto parse_layout(HirWord header) {
        return make_layout(header.header.data);
    }

    static uint64_t encode_header_data(size_t num_operands, Accessor<BitmapAttr> starred_bitmap) {
        CHECK(starred_bitmap.attr.num_bits == num_operands);
        return num_operands;
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             const BorrowedValue* operands,
                             size_t num_operands,
                             Accessor<BitmapAttr> starred_bitmap) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            CHECK(starred_bitmap.attr.num_bits == num_operands);
            Result<std::pair<const BorrowedValue*, size_t>> unpacked = unpack_starred_args(
                    starred_bitmap, operands, &ctx->runner->stack);
            if (!unpacked.is_ok()) return HirEvalStatus::Error;
            auto [items, num_items] = *unpacked;

            RefPtr<TupleValue> tuple = TupleValue::create_undefined(num_items);
            for (size_t i = 0; i < num_items; ++i)
                (*tuple)[i] = items[i].to_owned();
            *result = Value::from_tuple_value(std::move(tuple));
            return HirEvalStatus::OK;
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::Unpack> {
    static auto make_layout(uint32_t expected_len) {
        return make_opcode_layout(
            std::integral_constant<uint32_t, 1>{},  // iterable
            std::make_tuple(expected_len),
            {}
        );
    }

    static auto parse_layout(HirWord header) {
        return make_layout(header.header.data);
    }

    static uint64_t encode_header_data(uint32_t expected_len) {
        return expected_len;
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             BorrowedValue iterable,
                             uint32_t expected_len) {
        if (!iterable.is_tuple())
            return ctx->error(Error::kTypeError, "Expected a tuple, not ", use_type(iterable));

        size_t len = iterable.tuple_size();
        if (len != expected_len)
            return ctx->error(Error::kValueError, "Too ", len < expected_len ? "few" : "many",
                              " values to unpack (expected ", expected_len, ", got ", len, ")");

        // Return the input tuple. If we add support for additional iterables,
        // the idea is to convert them to a tuple here.
        *result = iterable.to_owned();
        return HirEvalStatus::OK;
    }
};

template <>
struct OpcodeImpl<HirOpcode::MakeClosure> {
    static auto make_layout(uint32_t num_default_values) {
        return make_opcode_layout<true>(
            num_default_values,  // default values of the parameters
            {},
            std::make_tuple(PyobjectAttr{})  // hir.Function
        );
    }

    static auto parse_layout(HirWord header) {
        return make_layout(header.header.data);
    }

    static uint64_t encode_header_data(size_t num_operands, Accessor<PyobjectAttr>) {
        return num_operands;
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             const BorrowedValue* operands,
                             size_t num_operands,
                             Accessor<PyobjectAttr> func_hir) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            PyPtr default_values = steal(PyTuple_New(num_operands));
            if (!default_values) return HirEvalStatus::Error;
            for (size_t i = 0; i < num_operands; ++i) {
                PyPtr var = value_to_ir_var(operands[i]);
                if (!var) return HirEvalStatus::Error;
                PyTuple_SET_ITEM(default_values.get(), i, var.release());
            }

            static LazyImporter make_closure_importer;
            PyObject* make_closure = make_closure_importer.get(
                    "cuda.tile._ir.core_ops", "make_closure");
            if (!make_closure) return HirEvalStatus::Error;
            PyPtr var = steal(PyObject_CallFunctionObjArgs(
                        make_closure, func_hir.get(), default_values.get(), loc, nullptr));
            if (!var) return HirEvalStatus::Error;
            Result<Value> r = value_from_ir_var(std::move(var));
            if (!r.is_ok()) return HirEvalStatus::Error;
            *result = std::move(*r);
            return HirEvalStatus::OK;
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::StaticEval> {
    static auto make_layout(uint32_t num_local_values) {
        return make_opcode_layout<true>(
            num_local_values,  // values of the local variables used by the expression
            {},
            std::make_tuple(PyobjectAttr{})  // hir.StaticEvalExpression
        );
    }

    static auto parse_layout(HirWord header) {
        return make_layout(header.header.data);
    }

    static uint64_t encode_header_data(size_t num_operands, Accessor<PyobjectAttr>) {
        return num_operands;
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             const BorrowedValue* operands,
                             size_t num_operands,
                             Accessor<PyobjectAttr> expr) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            PyPtr local_values = steal(PyTuple_New(num_operands));
            if (!local_values) return HirEvalStatus::Error;
            for (size_t i = 0; i < num_operands; ++i) {
                PyPtr var = value_to_ir_var(operands[i]);
                if (!var) return HirEvalStatus::Error;
                PyTuple_SET_ITEM(local_values.get(), i, var.release());
            }

            static LazyImporter do_static_eval_importer;
            PyObject* do_static_eval = do_static_eval_importer.get(
                    "cuda.tile._ir.static_eval_ops", "do_static_eval");
            if (!do_static_eval) return HirEvalStatus::Error;
            PyPtr var = steal(PyObject_CallFunctionObjArgs(
                        do_static_eval, expr.get(), local_values.get(), loc, nullptr));
            if (!var) return HirEvalStatus::Error;
            Result<Value> r = value_from_ir_var(std::move(var));
            if (!r.is_ok()) return HirEvalStatus::Error;
            *result = std::move(*r);
            return HirEvalStatus::OK;
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::StaticAssert> {
    static auto make_layout() {
        return make_opcode_layout<true>(
            std::integral_constant<uint32_t, 1>{},  // condition
            {},
            std::make_tuple(LabelAttr{})  // message_block, evaluated if the condition is false
        );
    }

    static auto parse_layout(HirWord) {
        return make_layout();
    }

    static uint64_t encode_header_data(Accessor<LabelAttr>) { return 0; }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             BorrowedValue condition,
                             Accessor<LabelAttr> message_block) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            PyPtr condition_var = value_to_ir_var(condition);
            if (!condition_var) return HirEvalStatus::Error;

            PyPtr message_label = hir_label_new(ctx->frame->program, message_block.get());
            if (!message_label) return HirEvalStatus::Error;

            static LazyImporter do_static_assert_importer;
            PyObject* do_static_assert = do_static_assert_importer.get(
                    "cuda.tile._ir.static_eval_ops", "do_static_assert");
            if (!do_static_assert) return HirEvalStatus::Error;
            PyPtr coro = steal(PyObject_CallFunctionObjArgs(
                        do_static_assert, condition_var.get(), message_label.get(), loc,
                        nullptr));
            if (!coro) return HirEvalStatus::Error;
            return ctx->delegate_to_python_coroutine(std::move(coro));
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::GeneratorYield> {
    static auto make_layout() {
        return make_opcode_layout(
            std::integral_constant<uint32_t, 1>{},  // the yielded value
            {},
            {}
        );
    }

    static auto parse_layout(HirWord) {
        return make_layout();
    }

    static uint64_t encode_header_data() { return 0; }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             BorrowedValue value) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            *result = value.to_owned();
            return HirEvalStatus::GeneratorYielded;
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::IfElse> {
    static auto make_layout() {
        return make_opcode_layout<true>(std::integral_constant<uint32_t, 1>{}, {},
                                  std::make_tuple(LabelAttr{}, LabelAttr{}));
    }

    static auto parse_layout(HirWord) {
        return make_layout();
    }

    static uint64_t encode_header_data(Accessor<LabelAttr>, Accessor<LabelAttr>) { return 0; }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             BorrowedValue cond,
                             Accessor<LabelAttr> then_block,
                             Accessor<LabelAttr> else_block) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            PyPtr cond_var = value_to_ir_var(cond);
            if (!cond_var) return HirEvalStatus::Error;

            PyPtr then_label = hir_label_new(ctx->frame->program, then_block.get());
            if (!then_label) return HirEvalStatus::Error;
            PyPtr else_label = hir_label_new(ctx->frame->program, else_block.get());
            if (!else_label) return HirEvalStatus::Error;

            static LazyImporter if_else_importer;
            PyObject* if_else = if_else_importer.get("cuda.tile._ir.control_flow_ops", "if_else");
            if (!if_else) return HirEvalStatus::Error;
            PyPtr coro = steal(PyObject_CallFunctionObjArgs(
                        if_else, cond_var.get(), then_label.get(), else_label.get(), loc,
                        nullptr));
            if (!coro) return HirEvalStatus::Error;
            return ctx->delegate_to_python_coroutine(std::move(coro));
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::TupleCompIf> {
    static auto make_layout() {
        return make_opcode_layout<true>(
            std::integral_constant<uint32_t, 1>{},  // cond (must be a constant)
            {},
            std::make_tuple(LabelAttr{})  // then_block
        );
    }

    static auto parse_layout(HirWord) {
        return make_layout();
    }

    static uint64_t encode_header_data(Accessor<LabelAttr>) { return 0; }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             BorrowedValue cond,
                             Accessor<LabelAttr> then_block) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            PyPtr cond_var = value_to_ir_var(cond);
            if (!cond_var) return HirEvalStatus::Error;

            PyPtr then_label = hir_label_new(ctx->frame->program, then_block.get());
            if (!then_label) return HirEvalStatus::Error;

            static LazyImporter tuple_comp_if_importer;
            PyObject* tuple_comp_if = tuple_comp_if_importer.get(
                    "cuda.tile._ir.control_flow_ops", "tuple_comp_if");
            if (!tuple_comp_if) return HirEvalStatus::Error;
            PyPtr coro = steal(PyObject_CallFunctionObjArgs(
                        tuple_comp_if, cond_var.get(), then_label.get(), loc, nullptr));
            if (!coro) return HirEvalStatus::Error;
            return ctx->delegate_to_python_coroutine(std::move(coro));
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::Loop> {
    static auto make_layout(bool has_iterable) {
        return make_opcode_layout<true>(static_cast<uint32_t>(has_iterable), {},
                                        std::make_tuple(LabelAttr{}));
    }

    static auto parse_layout(HirWord header) {
        return make_layout(header.header.data);
    }

    static uint64_t encode_header_data(size_t num_operands, Accessor<LabelAttr>) {
        CHECK(num_operands <= 1);
        return num_operands;
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             const BorrowedValue* operands,
                             size_t num_operands,
                             Accessor<LabelAttr> body) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            CHECK(num_operands <= 1);
            PyPtr iterable_var = num_operands ? value_to_ir_var(operands[0]) : newref(Py_None);
            if (!iterable_var) return HirEvalStatus::Error;

            PyPtr body_label = hir_label_new(ctx->frame->program, body.get());
            if (!body_label) return HirEvalStatus::Error;

            static LazyImporter loop_importer;
            PyObject* loop = loop_importer.get("cuda.tile._ir.control_flow_ops", "loop");
            if (!loop) return HirEvalStatus::Error;
            PyPtr coro = steal(PyObject_CallFunctionObjArgs(
                        loop, body_label.get(), iterable_var.get(), loc, nullptr));
            if (!coro) return HirEvalStatus::Error;
            return ctx->delegate_to_python_coroutine(std::move(coro));
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::StaticForeach> {
    static auto make_layout() {
        return make_opcode_layout<true>(
            std::integral_constant<uint32_t, 1>{},  // items (a tuple)
            {},
            std::make_tuple(LabelAttr{})  // body, evaluated once per item
        );
    }

    static auto parse_layout(HirWord) {
        return make_layout();
    }

    static uint64_t encode_header_data(Accessor<LabelAttr>) { return 0; }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             BorrowedValue items,
                             Accessor<LabelAttr> body) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            PyPtr items_var = value_to_ir_var(items);
            if (!items_var) return HirEvalStatus::Error;

            PyPtr body_label = hir_label_new(ctx->frame->program, body.get());
            if (!body_label) return HirEvalStatus::Error;

            static LazyImporter static_foreach_importer;
            PyObject* static_foreach = static_foreach_importer.get(
                    "cuda.tile._ir.static_eval_ops", "static_foreach");
            if (!static_foreach) return HirEvalStatus::Error;
            PyPtr coro = steal(PyObject_CallFunctionObjArgs(
                        static_foreach, body_label.get(), items_var.get(), loc, nullptr));
            if (!coro) return HirEvalStatus::Error;
            return ctx->delegate_to_python_coroutine(std::move(coro));
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::EndBranch> {
    static auto make_layout(bool with_result) {
        return make_opcode_layout<true>(static_cast<uint32_t>(with_result), {}, {});
    }

    static auto parse_layout(HirWord header) {
        return make_layout(header.header.data);
    }

    static uint64_t encode_header_data(size_t num_operands) {
        CHECK(num_operands <= 1);
        return num_operands;
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             const BorrowedValue* operands,
                             size_t num_operands) {
        if constexpr (PartialEval)
            return ctx->suspend();

        CHECK(num_operands <= 1);
        PyPtr output = num_operands ? value_to_ir_var(operands[0]) : newref(Py_None);
        if (!output) return HirEvalStatus::Error;

        static LazyImporter end_branch_importer;
        PyObject* end_branch = end_branch_importer.get("cuda.tile._ir.control_flow_ops",
                                                       "end_branch");
        if (!end_branch) return HirEvalStatus::Error;
        PyPtr res = steal(PyObject_CallFunctionObjArgs(end_branch, output.get(), loc, nullptr));
        if (!res) return HirEvalStatus::Error;
        *result = Value::none();
        return HirEvalStatus::OK;
    }
};


template <>
struct OpcodeImpl<HirOpcode::Continue> {
    static auto make_layout() {
        return make_opcode_layout<true>(std::integral_constant<uint32_t, 0>{}, {}, {});
    }

    static auto parse_layout(HirWord) {
        return make_layout();
    }

    static uint64_t encode_header_data() { return 0; }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx, Value* result, PyObject* loc) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            static LazyImporter continue_importer;
            PyObject* continue_ = continue_importer.get("cuda.tile._ir.control_flow_ops",
                                                        "continue_");
            if (!continue_) return HirEvalStatus::Error;
            PyPtr coro = steal(PyObject_CallOneArg(continue_, loc));
            if (!coro) return HirEvalStatus::Error;
            return ctx->delegate_to_python_coroutine(std::move(coro));
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::Break> {
    static auto make_layout() {
        return make_opcode_layout<true>(std::integral_constant<uint32_t, 0>{}, {}, {});
    }

    static auto parse_layout(HirWord) {
        return make_layout();
    }

    static uint64_t encode_header_data() { return 0; }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx, Value* result, PyObject* loc) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            static LazyImporter break_importer;
            PyObject* break_ = break_importer.get("cuda.tile._ir.control_flow_ops", "break_");
            if (!break_) return HirEvalStatus::Error;
            PyPtr coro = steal(PyObject_CallOneArg(break_, loc));
            if (!coro) return HirEvalStatus::Error;
            return ctx->delegate_to_python_coroutine(std::move(coro));
        }
    }
};

template <>
struct OpcodeImpl<HirOpcode::Return> {
    static auto make_layout(bool with_result) {
        return make_opcode_layout<true>(static_cast<uint32_t>(with_result), {}, {});
    }

    static auto parse_layout(HirWord header) {
        return make_layout(header.header.data);
    }

    static uint64_t encode_header_data(size_t num_operands) {
        CHECK(num_operands <= 1);
        return num_operands;
    }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             PyObject* loc,
                             const BorrowedValue* operands,
                             size_t num_operands) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            CHECK(num_operands <= 1);
            PyPtr value = num_operands ? value_to_ir_var(operands[0]) : newref(Py_None);
            if (!value) return HirEvalStatus::Error;

            static LazyImporter return_importer;
            PyObject* return_ = return_importer.get("cuda.tile._ir.control_flow_ops", "return_");
            if (!return_) return HirEvalStatus::Error;
            PyPtr coro = steal(PyObject_CallFunctionObjArgs(return_, value.get(), loc, nullptr));
            if (!coro) return HirEvalStatus::Error;
            return ctx->delegate_to_python_coroutine(std::move(coro));
        }
    }
};

// Ends the body of a helper function or a lambda, which evaluates to the given value
template <>
struct OpcodeImpl<HirOpcode::FunctionResult> {
    static auto make_layout() {
        return make_opcode_layout(
            std::integral_constant<uint32_t, 1>{},  // the function's result
            {},
            {}
        );
    }

    static auto parse_layout(HirWord) {
        return make_layout();
    }

    static uint64_t encode_header_data() { return 0; }

    template <bool PartialEval>
    static HirEvalStatus run(HirEvalContext<PartialEval>* ctx,
                             Value* result,
                             BorrowedValue value) {
        if constexpr (PartialEval) {
            return ctx->suspend();
        } else {
            *result = value.to_owned();
            return HirEvalStatus::FunctionFinished;
        }
    }
};

template <typename F>
static auto dispatch_instruction(HirWord** cursor, F&& func) {
    HirOpcode opcode = header_opcode(**cursor);
    switch (opcode) {
        #define DISPATCH_OPCODE(name) \
            case HirOpcode::name: return func(parse_instruction<HirOpcode::name>(cursor)); \

        FOREACH_HIR_OPCODE(DISPATCH_OPCODE)
        #undef DISPATCH_OPCODE

    case HirOpcode::BeginBlock: {
        // Header, end label
        *cursor += 2;
        return func(ParsedInstruction<HirOpcode::BeginBlock>{});
    }
    case HirOpcode::EndBlock: {
        // Header
        *cursor += 1;
        return func(ParsedInstruction<HirOpcode::EndBlock>{});
    }
    }
}

HirProgram::~HirProgram() {
    HirWord* cursor = data.data();
    for (uint32_t instruction_idx = 0; instruction_idx < instruction_count(); ++instruction_idx)
        dispatch_instruction(&cursor, [] (auto&& instr) { destroy_instruction(instr); });
}

// If `marked_instruction_id` is given, every line is prefixed with either "--> "
// (for the marked instruction) or four spaces.
static void print_block(HirProgram& program, InstructionPointer start, size_t indent,
                        std::optional<uint32_t> marked_instruction_id, StringBuilder* sb) {
    size_t depth = 0;
    uint32_t instruction_idx = start.index;
    HirWord* cursor = program.data.data() + start.offset;
    for (; instruction_idx < program.instruction_count(); ++instruction_idx) {
        HirOpcode opcode = header_opcode(cursor[0]);
        if (opcode == HirOpcode::EndBlock) {
            if (!depth) return;
            --depth;
        }

        if (marked_instruction_id)
            sb->append(instruction_idx == *marked_instruction_id ? "--> " : "    ");

        for (size_t i = 0; i < indent + depth; ++i)
            sb->append("  ");

        dispatch_instruction(&cursor, [instruction_idx, sb] (auto&& parsed_instruction) {
            print_instruction(parsed_instruction, instruction_idx, sb);
        });
        sb->append("\n");

        if (opcode == HirOpcode::BeginBlock)
            ++depth;
    }
}


struct HirBuilder {
    HirRunner runner;
    PyPtr partial_program;  // HirProgram
    Vec<uint32_t> begin_block_offsets;  // offsets of the open blocks' BeginBlock headers
    HirOpcode last_opcode = HirOpcode::BeginBlock;  // the program starts like a block

    template <HirOpcode Opcode, typename... Args>
    void append_instruction(PyObject* loc, Args&&... args) {
        HirProgram& prog = py_unwrap<HirProgram>(partial_program);
        using Layout = decltype(OpcodeImpl<Opcode>::parse_layout(HirWord{}));
        using NumOperands = decltype(Layout::num_operands);

        // Reconstruct the layout from the arguments
        NumOperands num_operands;
        if constexpr (!is_u32_constant_v<NumOperands>)
            num_operands = get_variadic_operands(args...).second;

        auto drop_operands_and_count = make_operand_and_count_dropper<NumOperands>();
        auto header_meta = drop_operands_and_count(
                [] (const auto&... rest) {
                    return TakeFirst<typename Layout::HeaderMetaTuple>()(rest...);
                },
                args...);

        auto drop_operands_and_header_meta = make_operand_and_header_meta_dropper(
                drop_operands_and_count, header_meta);
        auto attributes = drop_operands_and_header_meta(
                [] (const auto&... attr_args) { return std::make_tuple(attr_args.attr...); },
                args...);

        static_assert(std::is_same_v<decltype(Layout::header_meta), decltype(header_meta)>);
        static_assert(std::is_same_v<decltype(Layout::attributes), decltype(attributes)>);
        Layout layout{num_operands, std::move(header_meta), std::move(attributes)};

        // Allocate the required number of words
        size_t header_offset = prog.data.size();
        size_t attrs_offset = header_offset + 1;
        size_t operands_offset = attrs_offset + layout.attribute_words();
        prog.data.resize(operands_offset + layout.num_operands);

        // Put header, attributes and operands
        uint64_t header_data = make_operand_dropper<NumOperands>()(
                [&] (const auto&... a) { return OpcodeImpl<Opcode>::encode_header_data(a...); },
                args...);
        prog.data[header_offset] = make_header(Opcode, header_data);

        drop_operands_and_header_meta(
                [&] (const auto&... attr_args) {
                    layout.put_attributes(prog.data.data() + attrs_offset, attr_args...);
                }, args...);

        // Put the operands
        if constexpr (std::is_same_v<NumOperands, uint32_t>) {
            const BorrowedValue* operands = get_variadic_operands(args...).first;
            for (uint32_t i = 0; i < num_operands; ++i)
                prog.data[operands_offset + i].u64 = operands[i].to_owned().release_raw_data();
        } else {
            auto args_tup = std::forward_as_tuple(args...);
            static_for<NumOperands::value>([&] (auto i) {
                prog.data[operands_offset + i].u64
                    = std::get<i>(args_tup).to_owned().release_raw_data();
            });
        }
        prog.locations.push_back(newref(loc));
        last_opcode = Opcode;

        // Count Continue, Break and Return in the innermost block's BeginBlock header,
        // including those nested in the branches of an IfElse
        if (!begin_block_offsets.empty()) {
            HirWord& block_header = prog.data[begin_block_offsets.back()];
            if constexpr (Opcode == HirOpcode::Continue
                          || Opcode == HirOpcode::Break
                          || Opcode == HirOpcode::Return) {
                ++block_header.header.data;
            } else if constexpr (Opcode == HirOpcode::IfElse) {
                drop_operands_and_header_meta(
                        [&] (const Accessor<LabelAttr>& then_block,
                             const Accessor<LabelAttr>& else_block) {
                            block_header.header.data
                                += prog.begin_block_header(then_block.get()).header.data
                                + prog.begin_block_header(else_block.get()).header.data;
                        }, args...);
            }
        }
    }

    uint32_t append_begin_block(PyObject* loc) {
        HirProgram& prog = py_unwrap<HirProgram>(partial_program);
        uint32_t ret = prog.data.size();
        prog.data.push_back(make_header(HirOpcode::BeginBlock, 0));
        prog.data.push_back(HirWord{.iptr = {}});  // to be filled later
        prog.locations.push_back(newref(loc));
        last_opcode = HirOpcode::BeginBlock;
        return ret;
    }

    void append_end_block(PyObject* loc) {
        HirProgram& prog = py_unwrap<HirProgram>(partial_program);
        // Remember the block's last opcode in the EndBlock header
        uint64_t last = static_cast<uint64_t>(last_opcode);
        prog.data.push_back(make_header(HirOpcode::EndBlock, last));
        prog.locations.push_back(newref(loc));
        last_opcode = HirOpcode::EndBlock;
    }

    static PyTypeObject pytype;
};

template <HirOpcode Opcode, typename... Args>
HirEvalStatus HirEvalContext<true>::delegate_to(Value* result, Args&&... args) {
    HirEvalStatus status = OpcodeImpl<Opcode>::run(this, result, args...);
    if (status == HirEvalStatus::Suspended) {
        HirProgram& prog = py_unwrap<HirProgram>(builder->partial_program);
        uint32_t result_value_id = prog.instruction_count();
        if constexpr (kOpcodeRequiresLoc<Opcode>) {
            // The locations is already in the args
            builder->append_instruction<Opcode>(args...);
        } else {
            builder->append_instruction<Opcode>(cur_loc, args...);
        }
        *result = BorrowedValue::hir_value_id(result_value_id).to_owned();
        return HirEvalStatus::OK;
    }
    return status;
}


static Result<BorrowedValue> parse_operand(PyObject* obj) {
    if (Py_TYPE(obj) == &HirValue::pytype)
        return py_unwrap<HirValue>(obj).value.borrow();

    BorrowedValue value = borrowed_value_from_pyobject(obj);
    if (value.is_unsupported_pyobject())
        return raise(PyExc_TypeError, "Operand has unsupported type ", Py_TYPE(obj)->tp_name);

    return value;
}

// Unlike parse_operand(), requires the operand to be a HirValue
static Result<BorrowedValue> parse_hir_value(PyObject* obj, const char* name) {
    if (Py_TYPE(obj) != &HirValue::pytype)
        return raise(PyExc_TypeError, name, " must be a HirValue, not ", Py_TYPE(obj)->tp_name);
    return py_unwrap<HirValue>(obj).value.borrow();
}


struct HirStarred {
    Value value;
    static PyTypeObject pytype;
};

static PyObject* HirStarred_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    if (PyTuple_GET_SIZE(args) != 1 || (kwargs && PyDict_GET_SIZE(kwargs) != 0)) {
        raise(PyExc_TypeError, "HirStarred(value) expected 1 positional argument");
        return nullptr;
    }

    Result<BorrowedValue> value_res = parse_operand(PyTuple_GET_ITEM(args, 0));
    if (!value_res.is_ok()) return nullptr;

    PyPtr ret = py_create_object<HirStarred>(type);
    if (!ret) return nullptr;
    py_unwrap<HirStarred>(ret).value = value_res->to_owned();
    return ret.release();
}

PyTypeObject HirStarred::pytype = {
    .tp_name = "cuda.tile._cext.HirStarred",
    .tp_basicsize = sizeof(PythonWrapper<HirStarred>),
    .tp_dealloc = pywrapper_dealloc<HirStarred>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = HirStarred_new,
};

template <HirOpcode Opcode, typename... Args>
static PyPtr append_op(HirBuilder* builder, PyObject* loc, Args&&... args) {
    HirEvalContext<true> ctx = {&builder->runner, loc, builder};
    Value res;
    HirEvalStatus status;
    if constexpr (kOpcodeRequiresLoc<Opcode>)
        status = ctx.delegate_to<Opcode>(&res, loc, std::forward<Args>(args)...);
    else
        status = ctx.delegate_to<Opcode>(&res, std::forward<Args>(args)...);
    switch (status) {
    case HirEvalStatus::OK: return wrap_value(std::move(res), newref(loc));
    case HirEvalStatus::Error: return {};
    case HirEvalStatus::DelegatedToCoroutine: CHECK_UNREACHABLE;
    case HirEvalStatus::Suspended: CHECK_UNREACHABLE;
    case HirEvalStatus::GeneratorYielded: CHECK_UNREACHABLE;
    case HirEvalStatus::FunctionFinished: CHECK_UNREACHABLE;
    }
    CHECK_UNREACHABLE;
}

static PyObject* HirBuilder_call(PyObject* self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 4) {
        raise(PyExc_TypeError, "HirBuilder.call(callee, args, kwargs, loc) expected 4 arguments");
        return nullptr;
    }

    Result<BorrowedValue> callee_res = parse_operand(args[0]);
    if (!callee_res.is_ok()) return nullptr;

    PyObject* pos_args = args[1];
    if (!PyTuple_Check(pos_args)) {
        raise(PyExc_TypeError, "args must be a tuple, not ", Py_TYPE(pos_args)->tp_name);
        return nullptr;
    }
    size_t num_pos_args = PyTuple_GET_SIZE(pos_args);

    PyObject* kwargs = args[2];
    if (!PyTuple_Check(kwargs)) {
        raise(PyExc_TypeError, "kwargs must be a tuple, not ", Py_TYPE(kwargs)->tp_name);
        return nullptr;
    }

    size_t num_kwargs = PyTuple_GET_SIZE(kwargs);

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    builder.runner.stack.clear();
    auto [operands, num_operands, starred_bitmap, keywords]
            = allocate_operands_and_attributes<HirOpcode::GenericCall>(
                    &builder.runner.stack, num_pos_args, num_kwargs);
    CHECK(num_operands == 1 + num_pos_args + num_kwargs);

    new (&operands[0]) BorrowedValue(std::move(*callee_res));

    for (size_t i = 0; i < num_pos_args; ++i) {
        PyObject* arg = PyTuple_GET_ITEM(pos_args, i);
        if (Py_TYPE(arg) == &HirStarred::pytype) {
            const HirStarred& starred = py_unwrap<HirStarred>(arg);
            starred_bitmap->set(i);
            new (&operands[1 + i]) BorrowedValue(starred.value.borrow());
        } else {
            Result<BorrowedValue> r = parse_operand(arg);
            if (!r.is_ok()) return nullptr;
            new (&operands[1 + i]) BorrowedValue(*r);
        }
    }

    for (size_t i = 0; i < num_kwargs; ++i) {
        PyObject* pair = PyTuple_GET_ITEM(kwargs, i);
        if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) {
            raise(PyExc_TypeError, "Elements of `kwargs` must be length-2 tuples");
            return nullptr;
        }
        PyObject* keyword = PyTuple_GET_ITEM(pair, 0);
        if (keyword != Py_None) {
            if (!PyUnicode_CheckExact(keyword)) {
                raise(PyExc_TypeError, "Keywords must be strings, not ", Py_TYPE(keyword)->tp_name);
                return nullptr;
            }
            PyPtr keyword_ref = newref(keyword);
            pyunicode_intern_in_place(&keyword_ref);
            if (!PyUnicode_CHECK_INTERNED(keyword_ref.get())) {
                raise(PyExc_RuntimeError, "Failed to intern keyword string", keyword_ref.get());
                return nullptr;
            }
            keywords->set(i, std::move(keyword_ref));
        }

        Result<BorrowedValue> value_res = parse_operand(PyTuple_GET_ITEM(pair, 1));
        if (!value_res.is_ok()) return nullptr;
        new (&operands[1 + num_pos_args + i]) BorrowedValue(*value_res);
    }

    return append_op<HirOpcode::GenericCall>(
            &builder, args[3], operands, num_operands, *starred_bitmap, *keywords).release();
}

static PyObject* HirBuilder_build_formatted_string(PyObject* self, PyObject* const* args,
                                                   Py_ssize_t nargs) {
    if (nargs != 3) {
        raise(PyExc_TypeError,
              "HirBuilder.build_formatted_string(format, values, loc) expected 3 arguments");
        return nullptr;
    }

    PyObject* values = args[1];
    if (!PyTuple_Check(values)) {
        raise(PyExc_TypeError, "values must be a tuple, not ", Py_TYPE(values)->tp_name);
        return nullptr;
    }
    size_t num_values = PyTuple_GET_SIZE(values);

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    builder.runner.stack.clear();
    auto [operands, num_operands, format, value_locs]
            = allocate_operands_and_attributes<HirOpcode::BuildFormattedString>(
                    &builder.runner.stack, num_values);
    CHECK(num_operands == num_values);

    format->set(newref(args[0]));

    PyPtr locs = steal(PyTuple_New(num_values));
    if (!locs) return nullptr;
    for (size_t i = 0; i < num_values; ++i) {
        PyObject* value = PyTuple_GET_ITEM(values, i);
        Result<BorrowedValue> r = parse_hir_value(value, "Item of values");
        if (!r.is_ok()) return nullptr;
        new (&operands[i]) BorrowedValue(*r);
        PyTuple_SET_ITEM(locs.get(), i, Py_NewRef(py_unwrap<HirValue>(value).loc.get()));
    }
    value_locs->set(std::move(locs));

    return append_op<HirOpcode::BuildFormattedString>(
            &builder, args[2], operands, num_operands, *format, *value_locs).release();
}

static PyObject* HirBuilder_build_tuple(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        raise(PyExc_TypeError, "HirBuilder.build_tuple(items, loc) expected 2 arguments");
        return nullptr;
    }

    PyObject* items = args[0];
    if (!PyTuple_Check(items)) {
        raise(PyExc_TypeError, "items must be a tuple, not ", Py_TYPE(items)->tp_name);
        return nullptr;
    }
    size_t num_items = PyTuple_GET_SIZE(items);

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    builder.runner.stack.clear();

    auto [operands, num_operands, starred_bitmap]
            = allocate_operands_and_attributes<HirOpcode::BuildTuple>(
                    &builder.runner.stack, num_items);
    CHECK(num_operands == num_items);

    for (size_t i = 0; i < num_items; ++i) {
        PyObject* item = PyTuple_GET_ITEM(items, i);
        if (Py_TYPE(item) == &HirStarred::pytype) {
            starred_bitmap->set(i);
            new (&operands[i]) BorrowedValue(py_unwrap<HirStarred>(item).value.borrow());
        } else {
            Result<BorrowedValue> r = parse_hir_value(item, "Item of items");
            if (!r.is_ok()) return nullptr;
            new (&operands[i]) BorrowedValue(*r);
        }
    }

    return append_op<HirOpcode::BuildTuple>(
            &builder, args[1], operands, num_operands, *starred_bitmap).release();
}

static Result<InstructionPointer> parse_label(const HirBuilder& builder, PyObject* py_label,
                                              const char* name) {
    if (Py_TYPE(py_label) != &HirLabel::pytype) {
        return raise(PyExc_TypeError, name, " must be a HirLabel, not ",
                     Py_TYPE(py_label)->tp_name);
    }
    const HirLabel& label = py_unwrap<HirLabel>(py_label);
    if (label.program.get() != builder.partial_program.get())
        return raise(PyExc_ValueError, "Label from wrong program");
    if (!label.iptr.index)
        return raise(PyExc_ValueError, name, " must point to the start of a nested block");
    HirProgram& prog = py_unwrap<HirProgram>(label.program);
    HirOpcode last_opcode;
    if (!prog.get_block_last_opcode(label.iptr, &last_opcode))
        return raise(PyExc_ValueError, name, " must point to a closed block");
    return label.iptr;
}

static PyObject* HirBuilder_if_else(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 4) {
        raise(PyExc_TypeError,
              "HirBuilder.if_else(cond, then_block, else_block, loc) expected 4 arguments");
        return nullptr;
    }

    Result<BorrowedValue> cond_res = parse_hir_value(args[0], "cond");
    if (!cond_res.is_ok()) return nullptr;

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    Result<InstructionPointer> then_res = parse_label(builder, args[1], "then_block");
    if (!then_res.is_ok()) return nullptr;
    Result<InstructionPointer> else_res = parse_label(builder, args[2], "else_block");
    if (!else_res.is_ok()) return nullptr;

    builder.runner.stack.clear();
    auto [operands, num_operands, then_block, else_block]
            = allocate_operands_and_attributes<HirOpcode::IfElse>(&builder.runner.stack);
    then_block->set(*then_res);
    else_block->set(*else_res);

    return append_op<HirOpcode::IfElse>(
            &builder, args[3], *cond_res, *then_block, *else_block).release();
}

static PyObject* HirBuilder_tuple_comp_if(PyObject* self, PyObject* const* args,
                                          Py_ssize_t nargs) {
    if (nargs != 3) {
        raise(PyExc_TypeError,
              "HirBuilder.tuple_comp_if(cond, then_block, loc) expected 3 arguments");
        return nullptr;
    }

    Result<BorrowedValue> cond_res = parse_hir_value(args[0], "cond");
    if (!cond_res.is_ok()) return nullptr;

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    Result<InstructionPointer> then_res = parse_label(builder, args[1], "then_block");
    if (!then_res.is_ok()) return nullptr;

    builder.runner.stack.clear();
    auto [operands, num_operands, then_block]
            = allocate_operands_and_attributes<HirOpcode::TupleCompIf>(&builder.runner.stack);
    then_block->set(*then_res);

    return append_op<HirOpcode::TupleCompIf>(
            &builder, args[2], *cond_res, *then_block).release();
}

static PyObject* HirBuilder_loop(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2 && nargs != 3) {
        raise(PyExc_TypeError, "HirBuilder.loop(body, [iterable,] loc) expected 2 or 3 arguments");
        return nullptr;
    }
    bool has_iterable = nargs == 3;

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    Result<InstructionPointer> body_res = parse_label(builder, args[0], "body");
    if (!body_res.is_ok()) return nullptr;

    builder.runner.stack.clear();
    auto [operands, num_operands, body]
            = allocate_operands_and_attributes<HirOpcode::Loop>(&builder.runner.stack,
                                                                has_iterable);
    body->set(*body_res);

    if (has_iterable) {
        Result<BorrowedValue> iterable_res = parse_hir_value(args[1], "iterable");
        if (!iterable_res.is_ok()) return nullptr;
        new (&operands[0]) BorrowedValue(*iterable_res);
    }

    return append_op<HirOpcode::Loop>(
            &builder, args[nargs - 1], operands, num_operands, *body).release();
}

static PyObject* HirBuilder_static_foreach(PyObject* self, PyObject* const* args,
                                           Py_ssize_t nargs) {
    if (nargs != 3) {
        raise(PyExc_TypeError,
              "HirBuilder.static_foreach(body, items, loc) expected 3 arguments");
        return nullptr;
    }

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    Result<InstructionPointer> body_res = parse_label(builder, args[0], "body");
    if (!body_res.is_ok()) return nullptr;

    Result<BorrowedValue> items_res = parse_hir_value(args[1], "items");
    if (!items_res.is_ok()) return nullptr;

    builder.runner.stack.clear();
    auto [operands, num_operands, body]
            = allocate_operands_and_attributes<HirOpcode::StaticForeach>(&builder.runner.stack);
    body->set(*body_res);

    return append_op<HirOpcode::StaticForeach>(&builder, args[2], *items_res, *body).release();
}

static PyObject* HirBuilder_unpack(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        raise(PyExc_TypeError,
              "HirBuilder.unpack(iterable, expected_len, loc) expected 3 arguments");
        return nullptr;
    }

    Result<BorrowedValue> iterable_res = parse_hir_value(args[0], "iterable");
    if (!iterable_res.is_ok()) return nullptr;

    uint32_t expected_len = pylong_as<uint32_t>(args[1]);
    if (PyErr_Occurred()) return nullptr;

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    return append_op<HirOpcode::Unpack>(&builder, args[2], *iterable_res, expected_len)
            .release();
}

static PyObject* HirBuilder_make_closure(PyObject* self, PyObject* const* args,
                                         Py_ssize_t nargs) {
    if (nargs != 3) {
        raise(PyExc_TypeError,
              "HirBuilder.make_closure(func_hir, default_values, loc) expected 3 arguments");
        return nullptr;
    }

    PyObject* default_values = args[1];
    if (!PyTuple_Check(default_values)) {
        raise(PyExc_TypeError, "default_values must be a tuple, not ",
              Py_TYPE(default_values)->tp_name);
        return nullptr;
    }
    size_t num_default_values = PyTuple_GET_SIZE(default_values);

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    builder.runner.stack.clear();

    auto [operands, num_operands, func_hir]
            = allocate_operands_and_attributes<HirOpcode::MakeClosure>(
                    &builder.runner.stack, num_default_values);
    CHECK(num_operands == num_default_values);

    func_hir->set(newref(args[0]));

    for (size_t i = 0; i < num_default_values; ++i) {
        Result<BorrowedValue> r = parse_hir_value(PyTuple_GET_ITEM(default_values, i),
                                                  "Item of default_values");
        if (!r.is_ok()) return nullptr;
        new (&operands[i]) BorrowedValue(*r);
    }

    return append_op<HirOpcode::MakeClosure>(
            &builder, args[2], operands, num_operands, *func_hir).release();
}

static PyObject* HirBuilder_static_eval(PyObject* self, PyObject* const* args,
                                        Py_ssize_t nargs) {
    if (nargs != 3) {
        raise(PyExc_TypeError,
              "HirBuilder.static_eval(expr, local_values, loc) expected 3 arguments");
        return nullptr;
    }

    PyObject* local_values = args[1];
    if (!PyTuple_Check(local_values)) {
        raise(PyExc_TypeError, "local_values must be a tuple, not ",
              Py_TYPE(local_values)->tp_name);
        return nullptr;
    }
    size_t num_local_values = PyTuple_GET_SIZE(local_values);

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    builder.runner.stack.clear();

    auto [operands, num_operands, expr]
            = allocate_operands_and_attributes<HirOpcode::StaticEval>(
                    &builder.runner.stack, num_local_values);
    CHECK(num_operands == num_local_values);

    expr->set(newref(args[0]));

    for (size_t i = 0; i < num_local_values; ++i) {
        Result<BorrowedValue> r = parse_hir_value(PyTuple_GET_ITEM(local_values, i),
                                                  "Item of local_values");
        if (!r.is_ok()) return nullptr;
        new (&operands[i]) BorrowedValue(*r);
    }

    return append_op<HirOpcode::StaticEval>(
            &builder, args[2], operands, num_operands, *expr).release();
}

static PyObject* HirBuilder_static_assert(PyObject* self, PyObject* const* args,
                                          Py_ssize_t nargs) {
    if (nargs != 3) {
        raise(PyExc_TypeError,
              "HirBuilder.static_assert(condition, message_block, loc) expected 3 arguments");
        return nullptr;
    }

    Result<BorrowedValue> condition_res = parse_hir_value(args[0], "condition");
    if (!condition_res.is_ok()) return nullptr;

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    Result<InstructionPointer> message_res = parse_label(builder, args[1], "message_block");
    if (!message_res.is_ok()) return nullptr;

    builder.runner.stack.clear();
    auto [operands, num_operands, message_block]
            = allocate_operands_and_attributes<HirOpcode::StaticAssert>(&builder.runner.stack);
    message_block->set(*message_res);

    return append_op<HirOpcode::StaticAssert>(
            &builder, args[2], *condition_res, *message_block).release();
}

static PyObject* HirBuilder_generator_yield(PyObject* self, PyObject* const* args,
                                            Py_ssize_t nargs) {
    if (nargs != 2) {
        raise(PyExc_TypeError, "HirBuilder.generator_yield(value, loc) expected 2 arguments");
        return nullptr;
    }

    Result<BorrowedValue> value_res = parse_hir_value(args[0], "value");
    if (!value_res.is_ok()) return nullptr;

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    return append_op<HirOpcode::GeneratorYield>(&builder, args[1], *value_res).release();
}

static PyObject* HirBuilder_function_result(PyObject* self, PyObject* const* args,
                                             Py_ssize_t nargs) {
    if (nargs != 2) {
        raise(PyExc_TypeError, "HirBuilder.function_result(value, loc) expected 2 arguments");
        return nullptr;
    }

    Result<BorrowedValue> value_res = parse_hir_value(args[0], "value");
    if (!value_res.is_ok()) return nullptr;

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    PyPtr res = append_op<HirOpcode::FunctionResult>(&builder, args[1], *value_res);
    if (!res) return nullptr;
    return Py_NewRef(Py_None);
}

static PyObject* HirBuilder_load_global(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        raise(PyExc_TypeError, "HirBuilder.load_global(value, loc) expected 2 arguments");
        return nullptr;
    }

    // Unlike parse_operand(), allow unsupported Python objects:
    // LoadGlobal converts them at runtime.
    BorrowedValue value = borrowed_value_from_pyobject(args[0]);

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    return append_op<HirOpcode::LoadGlobal>(&builder, args[1], value).release();
}

static PyObject* HirBuilder_load_undefined_variable(PyObject* self, PyObject* const* args,
                                                    Py_ssize_t nargs) {
    if (nargs != 2) {
        raise(PyExc_TypeError,
              "HirBuilder.load_undefined_variable(name, loc) expected 2 arguments");
        return nullptr;
    }

    PyObject* name = args[0];
    if (!PyUnicode_CheckExact(name)) {
        raise(PyExc_TypeError, "name must be a str, not ", Py_TYPE(name)->tp_name);
        return nullptr;
    }

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    builder.runner.stack.clear();

    auto [operands, num_operands, name_attr]
            = allocate_operands_and_attributes<HirOpcode::LoadUndefinedVariable>(
                    &builder.runner.stack);
    name_attr->set(newref(name));

    return append_op<HirOpcode::LoadUndefinedVariable>(&builder, args[1], *name_attr).release();
}

static PyObject* HirBuilder_store_local(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        raise(PyExc_TypeError, "HirBuilder.store_local(index, value, loc) expected 3 arguments");
        return nullptr;
    }

    uint32_t index = pylong_as<uint32_t>(args[0]);
    if (PyErr_Occurred()) return nullptr;

    Result<BorrowedValue> value_res = parse_hir_value(args[1], "value");
    if (!value_res.is_ok()) return nullptr;

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    PyPtr res = append_op<HirOpcode::StoreLocal>(&builder, args[2], *value_res, index);
    if (!res) return nullptr;
    return Py_NewRef(Py_None);
}

// For EndBranch and Return. `value` is nullable.
template <HirOpcode Opcode>
static PyObject* append_jump(PyObject* self, PyObject* value, PyObject* loc) {
    BorrowedValue operand = BorrowedValue::undefined();
    if (value) {
        Result<BorrowedValue> r = parse_hir_value(value, "value");
        if (!r.is_ok()) return nullptr;
        operand = *r;
    }

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    uint32_t num_operands = value ? 1 : 0;
    PyPtr res = append_op<Opcode>(&builder, loc, &operand, num_operands);
    if (!res) return nullptr;
    return Py_NewRef(Py_None);
}

static PyObject* HirBuilder_end_branch(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1 && nargs != 2) {
        raise(PyExc_TypeError, "HirBuilder.end_branch([value,] loc) expected 1 or 2 arguments");
        return nullptr;
    }
    PyObject* value = nargs == 2 ? args[0] : nullptr;
    return append_jump<HirOpcode::EndBranch>(self, value, args[nargs - 1]);
}

static PyObject* HirBuilder_continue(PyObject* self, PyObject* loc) {
    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    PyPtr res = append_op<HirOpcode::Continue>(&builder, loc);
    if (!res) return nullptr;
    return Py_NewRef(Py_None);
}

static PyObject* HirBuilder_break(PyObject* self, PyObject* loc) {
    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    PyPtr res = append_op<HirOpcode::Break>(&builder, loc);
    if (!res) return nullptr;
    return Py_NewRef(Py_None);
}

static PyObject* HirBuilder_return(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1 && nargs != 2) {
        raise(PyExc_TypeError, "HirBuilder.return_([value,] loc) expected 1 or 2 arguments");
        return nullptr;
    }
    PyObject* value = nargs == 2 ? args[0] : nullptr;
    return append_jump<HirOpcode::Return>(self, value, args[nargs - 1]);
}

static PyObject* HirBuilder_load_local(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        raise(PyExc_TypeError, "HirBuilder.load_local(depth, index, loc) expected 3 arguments");
        return nullptr;
    }

    uint32_t depth = pylong_as<uint32_t>(args[0]);
    if (PyErr_Occurred()) return nullptr;
    uint32_t index = pylong_as<uint32_t>(args[1]);
    if (PyErr_Occurred()) return nullptr;

    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    return append_op<HirOpcode::LoadLocal>(&builder, args[2], depth, index).release();
}

static bool is_jump_opcode(HirOpcode opcode) {
    switch (opcode) {
    case HirOpcode::GenericCall:
    case HirOpcode::LoadLocal:
    case HirOpcode::StoreLocal:
    case HirOpcode::LoadGlobal:
    case HirOpcode::LoadUndefinedVariable:
    case HirOpcode::BuildFormattedString:
    case HirOpcode::BuildTuple:
    case HirOpcode::Unpack:
    case HirOpcode::MakeClosure:
    case HirOpcode::StaticEval:
    case HirOpcode::StaticAssert:
    case HirOpcode::GeneratorYield:
    case HirOpcode::IfElse:
    case HirOpcode::TupleCompIf:
    case HirOpcode::Loop:
    case HirOpcode::StaticForeach:
    case HirOpcode::BeginBlock:
    case HirOpcode::EndBlock:
        return false;
    case HirOpcode::EndBranch:
    case HirOpcode::Continue:
    case HirOpcode::Break:
    case HirOpcode::Return:
    case HirOpcode::FunctionResult:
        return true;
    }
    CHECK_UNREACHABLE;
}

static PyObject* HirBuilder_is_ending_with_jump(PyObject* self, PyObject*) {
    return PyBool_FromLong(is_jump_opcode(py_unwrap<HirBuilder>(self).last_opcode));
}

static PyObject* HirBuilder_start_label(PyObject* self, PyObject*) {
    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    return hir_label_new(builder.partial_program, InstructionPointer{0, 0}).release();
}

static PyObject* HirBuilder_begin_block(PyObject* self, PyObject* loc) {
    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    uint32_t header_offset = builder.append_begin_block(loc);
    builder.begin_block_offsets.push_back(header_offset);
    HirProgram& prog = py_unwrap<HirProgram>(builder.partial_program);
    return hir_label_new(builder.partial_program, prog.end_iptr()).release();
}

static PyObject* HirBuilder_end_block(PyObject* self, PyObject* loc) {
    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    if (builder.begin_block_offsets.empty()) {
        raise(PyExc_ValueError, "There is no open block to be closed");
        return nullptr;
    }

    // Push EndBlock
    builder.append_end_block(loc);
    HirProgram& p = py_unwrap<HirProgram>(builder.partial_program);

    // Patch the end label in the BeginBlock
    uint32_t header_offset = builder.begin_block_offsets.back();
    builder.begin_block_offsets.pop_back();
    p.data[header_offset + 1].iptr = p.end_iptr();

    return Py_NewRef(Py_None);
}

static PyObject* HirBuilder_build(PyObject* self, PyObject*) {
    HirBuilder& builder = py_unwrap<HirBuilder>(self);
    if (!builder.begin_block_offsets.empty()) {
        raise(PyExc_ValueError, "Cannot build a program with unfinished block");
        return nullptr;
    }

    // Start over with an empty program, so that the builder stays usable
    PyPtr new_program = py_create_object<HirProgram>();
    if (!new_program) return nullptr;
    HirProgram& old_prog = py_unwrap<HirProgram>(builder.partial_program);
    py_unwrap<HirProgram>(new_program).loc = old_prog.loc;
    builder.last_opcode = HirOpcode::BeginBlock;
    return std::exchange(builder.partial_program, std::move(new_program)).release();
}

static PyMethodDef HirBuilder_methods[] = {
    {"call", reinterpret_cast<PyCFunction>(HirBuilder_call), METH_FASTCALL, nullptr},
    {"build_formatted_string", reinterpret_cast<PyCFunction>(HirBuilder_build_formatted_string),
     METH_FASTCALL, nullptr},
    {"build_tuple", reinterpret_cast<PyCFunction>(HirBuilder_build_tuple), METH_FASTCALL, nullptr},
    {"unpack", reinterpret_cast<PyCFunction>(HirBuilder_unpack), METH_FASTCALL, nullptr},
    {"make_closure", reinterpret_cast<PyCFunction>(HirBuilder_make_closure), METH_FASTCALL,
     nullptr},
    {"static_eval", reinterpret_cast<PyCFunction>(HirBuilder_static_eval), METH_FASTCALL,
     nullptr},
    {"static_assert", reinterpret_cast<PyCFunction>(HirBuilder_static_assert), METH_FASTCALL,
     nullptr},
    {"generator_yield", reinterpret_cast<PyCFunction>(HirBuilder_generator_yield),
     METH_FASTCALL, nullptr},
    {"function_result", reinterpret_cast<PyCFunction>(HirBuilder_function_result),
     METH_FASTCALL, nullptr},
    {"if_else", reinterpret_cast<PyCFunction>(HirBuilder_if_else), METH_FASTCALL, nullptr},
    {"tuple_comp_if", reinterpret_cast<PyCFunction>(HirBuilder_tuple_comp_if), METH_FASTCALL,
     nullptr},
    {"loop", reinterpret_cast<PyCFunction>(HirBuilder_loop), METH_FASTCALL, nullptr},
    {"static_foreach", reinterpret_cast<PyCFunction>(HirBuilder_static_foreach), METH_FASTCALL,
     nullptr},
    {"store_local", reinterpret_cast<PyCFunction>(HirBuilder_store_local), METH_FASTCALL,
     nullptr},
    {"load_global", reinterpret_cast<PyCFunction>(HirBuilder_load_global), METH_FASTCALL,
     nullptr},
    {"load_undefined_variable", reinterpret_cast<PyCFunction>(HirBuilder_load_undefined_variable),
     METH_FASTCALL, nullptr},
    {"end_branch", reinterpret_cast<PyCFunction>(HirBuilder_end_branch), METH_FASTCALL, nullptr},
    {"continue_", HirBuilder_continue, METH_O, nullptr},
    {"break_", HirBuilder_break, METH_O, nullptr},
    {"return_", reinterpret_cast<PyCFunction>(HirBuilder_return), METH_FASTCALL, nullptr},
    {"load_local", reinterpret_cast<PyCFunction>(HirBuilder_load_local), METH_FASTCALL, nullptr},
    {"is_ending_with_jump", HirBuilder_is_ending_with_jump, METH_NOARGS, nullptr},
    {"start_label", HirBuilder_start_label, METH_NOARGS, nullptr},
    {"begin_block", HirBuilder_begin_block, METH_O, nullptr},
    {"end_block", HirBuilder_end_block, METH_O, nullptr},
    {"build", HirBuilder_build, METH_NOARGS, nullptr},
    {}
};

static PyObject* HirBuilder_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    if (PyTuple_GET_SIZE(args) != 1 || (kwargs && PyDict_GET_SIZE(kwargs) != 0)) {
        raise(PyExc_TypeError, "HirBuilder(loc) expected 1 positional argument");
        return nullptr;
    }

    PyPtr program = py_create_object<HirProgram>();
    if (!program) return nullptr;
    py_unwrap<HirProgram>(program).loc = newref(PyTuple_GET_ITEM(args, 0));

    PyPtr ret = py_create_object<HirBuilder>(type);
    if (!ret) return nullptr;
    py_unwrap<HirBuilder>(ret).partial_program = std::move(program);
    return ret.release();
}

PyTypeObject HirBuilder::pytype = {
    .tp_name = "cuda.tile._cext.HirBuilder",
    .tp_basicsize = sizeof(PythonWrapper<HirBuilder>),
    .tp_dealloc = pywrapper_dealloc<HirBuilder>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = HirBuilder_methods,
    .tp_new = HirBuilder_new,
};

static PyObject* py_hir_eval_block(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2 && nargs != 3) {
        raise(PyExc_TypeError,
              "hir_eval_block(frame, label, [block_parameter]) expected 2 or 3 arguments");
        return nullptr;
    }

    PyObject* py_frame = args[0];
    if (Py_TYPE(py_frame) != &HirFrame::pytype) {
        raise(PyExc_TypeError, "frame must be a HirFrame, not ", Py_TYPE(py_frame)->tp_name);
        return nullptr;
    }

    PyObject* py_label = args[1];
    if (Py_TYPE(py_label) != &HirLabel::pytype) {
        raise(PyExc_TypeError, "label must be a HirLabel, not ", Py_TYPE(py_label)->tp_name);
        return nullptr;
    }

    HirFrame* frame = reinterpret_cast<HirFrame*>(py_frame);
    const HirLabel& label = py_unwrap<HirLabel>(py_label);

    if (frame->program != label.program) {
        raise(PyExc_ValueError, "Block's program is not matching the current frame!");
        return nullptr;
    }

    Value block_parameter;
    if (nargs == 3) {
        if (!label.iptr.index) {
            raise(PyExc_ValueError, "Label 0 does not point to start of a block");
            return nullptr;
        }
        Result<Value> param = value_from_ir_var(newref(args[2]));
        if (!param.is_ok()) return nullptr;
        block_parameter = std::move(*param);
    }

    return eval_block_awaitable_new(frame, label.iptr, std::move(block_parameter)).release();
}

static PyMethodDef hir_functions[] = {
    {"hir_run", reinterpret_cast<PyCFunction>(py_hir_run), METH_FASTCALL, nullptr},
    {"hir_eval_block", reinterpret_cast<PyCFunction>(py_hir_eval_block), METH_FASTCALL, nullptr},
    {}
};

Status hir_init(PyObject* m) {
    if (PyModule_AddFunctions(m, hir_functions) < 0)
        return ErrorRaised;

    if (PyType_Ready(&HirFrame::pytype) < 0)
        return ErrorRaised;
    if (PyModule_AddObjectRef(m, "HirFrame", reinterpret_cast<PyObject*>(&HirFrame::pytype)) < 0)
        return ErrorRaised;

    if (PyType_Ready(&HirLabel::pytype) < 0)
        return ErrorRaised;
    if (PyModule_AddObjectRef(m, "HirLabel", reinterpret_cast<PyObject*>(&HirLabel::pytype)) < 0)
        return ErrorRaised;

    if (PyType_Ready(&EvalBlockAwaitable::pytype) < 0)
        return ErrorRaised;

    if (PyType_Ready(&HirGeneratorYielded::pytype) < 0)
        return ErrorRaised;
    if (PyModule_AddObjectRef(m, "HirGeneratorYielded",
                              reinterpret_cast<PyObject*>(&HirGeneratorYielded::pytype)) < 0)
        return ErrorRaised;

    if (PyType_Ready(&HirProgram::pytype) < 0)
        return ErrorRaised;
    if (PyModule_AddObjectRef(m, "HirProgram",
                reinterpret_cast<PyObject*>(&HirProgram::pytype)) < 0)
        return ErrorRaised;

    if (PyType_Ready(&HirValue::pytype) < 0)
        return ErrorRaised;
    if (PyModule_AddObjectRef(m, "HirValue", reinterpret_cast<PyObject*>(&HirValue::pytype)) < 0)
        return ErrorRaised;

    if (PyType_Ready(&HirStarred::pytype) < 0)
        return ErrorRaised;
    if (PyModule_AddObjectRef(m, "HirStarred",
                reinterpret_cast<PyObject*>(&HirStarred::pytype)) < 0)
        return ErrorRaised;

    if (PyType_Ready(&HirBuilder::pytype) < 0)
        return ErrorRaised;
    if (PyModule_AddObjectRef(m, "HirBuilder",
                reinterpret_cast<PyObject*>(&HirBuilder::pytype)) < 0)
        return ErrorRaised;

    if (PyType_Ready(&HirBlockTerminated_pytype) < 0)
        return ErrorRaised;
    if (PyModule_AddObjectRef(m, "HirBlockTerminated",
                reinterpret_cast<PyObject*>(&HirBlockTerminated_pytype)) < 0)
        return ErrorRaised;
    g_hir_block_terminated = PyObject_New(PyObject, &HirBlockTerminated_pytype);
    if (!g_hir_block_terminated)
        return ErrorRaised;
    if (PyModule_AddObjectRef(m, "HIR_BLOCK_TERMINATED", g_hir_block_terminated) < 0)
        return ErrorRaised;

    return OK;
}



