//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file contains evaluation/folding implementations for KGEN attributes.
// These methods implement
// ContextuallyEvaluatedAttrInterface::evaluateWithContext.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/FoldUtils.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ParamListReduceAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ParamListReduceAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto paramList = sugarDynCast<ParamListAttr>(getParamList());
  auto reducer = sugarDynCast<GeneratorAttr>(getGenerator());

  if (!paramList || !reducer)
    return failure();

  // We have a concrete value for both the generator/variadic, then fold
  unsigned eltCnt = paramList.getValues().size();
  TypedAttr reducedVal = sugarCast<TypedAttr>(getBase());
  for (unsigned i = 0; i < eltCnt; ++i) {
    IntegerAttr vaIdx =
        IntegerAttr::get(IndexType::get(paramList.getContext()), i);
    GeneratorAttr spGen = reducer.getSpecializedGenerator(
        {reducedVal, paramList, vaIdx}, &context);
    if (!spGen)
      return TypedAttr();
    // This should never happen, we should have verified VariadicMapAttr.
    assert(spGen.isFullyBound() && "invalid form of variadic map");
    reducedVal = spGen.getInstantiatedValue();
  }

  return {reducedVal};
}

//===----------------------------------------------------------------------===//
// ParamListTabulateAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ParamListTabulateAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto cntAttr = sugarDynCast<IntegerAttr>(getCount());
  auto genAttr = sugarDynCast<GeneratorAttr>(getGenerator());
  if (!cntAttr || !genAttr)
    return failure();

  int64_t n = cntAttr.getInt();
  if (n < 0)
    return failure();

  SmallVector<TypedAttr> values;
  values.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    IntegerAttr idxAttr = IntegerAttr::get(IndexType::get(getContext()), i);
    GeneratorAttr spGen = genAttr.getSpecializedGenerator({idxAttr}, &context);
    if (!spGen)
      return TypedAttr();
    if (!spGen.isFullyBound())
      return failure();
    values.push_back(sugarCast<TypedAttr>(spGen.getInstantiatedValue()));
  }
  return {ParamListAttr::get(values, getType())};
}

//===----------------------------------------------------------------------===//
// GetWitnessAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
GetWitnessAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  // Returns nullopt when `handle` has no matching conformance.
  auto simplifyFrom =
      [&](ResolvedStructHandle handle) -> std::optional<FailureOr<TypedAttr>> {
    Operation *conformanceOp =
        context.resolveConformanceForStruct(handle, getTraitSymbol());
    if (!conformanceOp)
      return std::nullopt;

    FailureOr<TypedAttr> result = failure();
    context.withEvaluator(handle.decl.getInputParams(), handle.paramValues,
                          [&](ParameterEvaluator &evaluator) {
                            result = simplify(
                                cast<ConformanceOp>(conformanceOp), &evaluator);
                          });
    if (failed(result))
      context.emitMaterializationError(
          "failed to locate witness entry '" + getWitnessName().getValue() +
          "' for trait '" + getTraitSymbol().getFlattenedName().getValue() +
          "'");
    return result;
  };

  // First, look for the requested conformance directly on the anchor type.
  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (succeeded(resolvedOr)) {
    if (auto result = simplifyFrom(*resolvedOr))
      return *result;
  }

  // The anchor type does not itself carry the requested conformance, or is not
  // yet resolvable. Check the extension conformance table.
  if (auto extension =
          sugarDynCast<ExtensionAttr>(SugarAttr::strip(getTypeValue()))) {
    for (TypedAttr ext : extension.getExtensions()) {
      FailureOr<ResolvedStructHandle> extResolvedOr =
          context.resolveStructOp(ext, /*acceptAsync=*/false);
      if (failed(extResolvedOr))
        return failure();
      if (auto result = simplifyFrom(*extResolvedOr))
        return *result;
    }
  }

  // Neither the anchor nor any extension provided the conformance. If the
  // anchor was resolvable and we still could not evaluate this is an error.
  if (succeeded(resolvedOr))
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolvedOr->decl.getOperation()).getValue() +
        "' does not have witness table for trait '" +
        getTraitSymbol().getFlattenedName().getValue() + "'");
  return failure();
}

//===----------------------------------------------------------------------===//
// ParamOperatorAttr
//===----------------------------------------------------------------------===//

// FIXME(MOCO-4110): The reason why we need to extend
// `ParamOperatorAttr::evaluateWithContext` here is because
// `ContextuallyEvaluatedAttrInterface` is **NOT** always created with a
// context. For things like `TypeConformsToTraitAttr` that canonicalizes to a
// `ParamOperatorAttr` without a context, it then needs to be further evaluated
// here.
//
// If we can make `EvalContext` mandatory for creating any
// `ContextuallyEvaluatedAttrInterface`, we could then delete the code below. At
// that time, every `ContextuallyEvaluatedAttrInterface` should be in
// canonicalized + fully-evaluated form UPON CONSTRUCTION, which eliminates the
// need for `evaluator.getReboundXXX`, and the attr evaluation can be
// implemented in a much more local way.
FailureOr<TypedAttr> ParamOperatorAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  bool changed = false;
  SmallVector<TypedAttr> operands(getOperands());
  for (auto [i, cur] : llvm::enumerate(operands)) {
    if (auto ctxEval = sugarDynCast<ContextuallyEvaluatedAttrInterface>(cur)) {
      FailureOr<TypedAttr> result = context.evaluateExpression(ctxEval);
      if (failed(result)) {
        operands[i] = cur;
        continue;
      }

      TypedAttr evaluated = *result;
      // Defer the entire evaluation
      if (!evaluated)
        return TypedAttr();

      changed |= (operands[i] != evaluated);
      operands[i] =
          ParamOperatorAttr::getRebind(evaluated, operands[i].getType());
    }
  }
  if (changed)
    return ParamOperatorAttr::get(getOpcode(), operands, getType());

  // Can not be further folded.
  return failure();
}

//===----------------------------------------------------------------------===//
// ParamIdenticalAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ParamIdenticalAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // `get()` decided everything that does not need a target, so the only thing
  // left to contribute here is the index bit width.
  TargetInfoAttr target = context.getTargetInfo();
  if (!target)
    return failure();

  // Given the index bit width, an operand's index-like leaves re-expressed at
  // that width are canonical for the value, so two operands denote the same
  // value exactly when that key is the same uniqued attribute and one pass
  // answers for the whole class.
  //
  // Only an operand that is fully evaluated and free of unknowns has a value
  // for a key to stand for; the rest sit out the partition. Two others can
  // still settle the class `false` around one, which is why this counts
  // distinct keys rather than requiring them all to match.
  DenseSet<Attribute> keys;
  bool sawResidual = false;
  for (TypedAttr operand : getOperands()) {
    TypedAttr canonical = getCanonicalAttr(operand);
    PreparedConstant prepared(canonical, target);
    if (!ParameterAttr::isSimpleConstant(canonical) || prepared.hasUnknown()) {
      sawResidual = true;
      continue;
    }
    // One pair that cannot denote the same value settles the whole class, so
    // stop before keying the rest.
    keys.insert(prepared.getKey());
    if (keys.size() > 1)
      return TypedAttr(SIMDAttr::getScalarBool(getContext(), false));
  }

  // An operand with no key never collapses into another, so it leaves the class
  // residual however the rest of them compare.
  if (sawResidual)
    return failure();

  assert(!keys.empty() && "an identity class holds at least two operands");
  return TypedAttr(SIMDAttr::getScalarBool(getContext(), true));
}

//===----------------------------------------------------------------------===//
// TypeConformsToTraitAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> TypeConformsToTraitAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    if (context.isMaterializationContext()) {
      if (auto typeParam = dyn_cast<TypeParamAttr>(getTypeValue());
          typeParam && !isa<TypeValueType>(typeParam.getTypeValue())) {
        // This is a non-struct type.
        //
        // FIXME: non-struct type is TRP, maybe we should resolve it to
        // `__MLIRType` instead?
        return TypedAttr(SIMDAttr::getScalarBool(getContext(), false));
      }
    }
    return failure();
  }

  ResolvedStructHandle resolved = *resolvedOr;
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                        [&](ParameterEvaluator &evaluator) {
                          result = simplify(resolved.decl, evaluator);
                        });
  return result;
}

//===----------------------------------------------------------------------===//
// Struct Field Attr evaluateWithContext implementations
//===----------------------------------------------------------------------===//

/// Compute the byte offset of a field within a struct given its field types.
/// Returns failure if the field index is out of bounds or if size/alignment
/// cannot be determined for any field type.
static FailureOr<int64_t>
computeStructFieldOffset(ArrayRef<Type> fieldTypes, int64_t fieldIndex,
                         TargetInfoAttr target,
                         llvm::function_ref<void(const Twine &)> emitError) {
  if (fieldIndex < 0 || fieldIndex >= static_cast<int64_t>(fieldTypes.size())) {
    emitError("field index " + std::to_string(fieldIndex) +
              " is out of bounds for struct with " +
              std::to_string(fieldTypes.size()) + " fields");
    return failure();
  }

  int64_t offset = 0;
  for (int64_t i = 0; i < fieldIndex; ++i) {
    std::optional<int64_t> curFieldAlign =
        DataLayoutInterface::getTypeABIAlign(target, fieldTypes[i]);
    std::optional<int64_t> curFieldSize =
        DataLayoutInterface::getTypeAllocSize(target, fieldTypes[i]);
    if (!curFieldAlign || !curFieldSize) {
      emitError("could not determine size or alignment for field type");
      return failure();
    }
    offset = llvm::alignTo(offset, *curFieldAlign) + *curFieldSize;
  }

  // Align to the target field's alignment.
  std::optional<int64_t> fieldAlign =
      DataLayoutInterface::getTypeABIAlign(target, fieldTypes[fieldIndex]);
  if (!fieldAlign) {
    emitError("could not determine alignment for field type");
    return failure();
  }
  return llvm::alignTo(offset, *fieldAlign);
}

/// Extract field types from a struct, wrapping each in ParamType.
static SmallVector<Type>
getFieldTypesFromStruct(StructInstanceType structType) {
  SmallVector<Type> fieldTypes;
  for (StructDefFieldAttr field : structType.getFields())
    fieldTypes.push_back(ParamType::get(field.getTypeValue()));
  return fieldTypes;
}

/// Extract and rebind field types from a struct declaration using an evaluator.
/// Returns an empty vector for empty structs.
///
/// Returns std::nullopt when a field type is not ready yet. A null result from
/// `getReboundAttribute` is the async retry signal, so callers must forward it
/// as a null success rather than collapsing it into failure().
static std::optional<SmallVector<Type>>
rebindFieldTypes(StructDeclInterface decl, ParameterEvaluator &evaluator) {
  SmallVector<TypedAttr> fieldTypeAttrs;
  // MetaType does not really matter here, they will be striped later by
  // `ParamType::get(rebound)` anyway.
  decl.getFieldTypes(fieldTypeAttrs, TypeType::get(decl.getContext()));

  SmallVector<Type> fieldTypes;
  for (TypedAttr typeAttr : fieldTypeAttrs) {
    TypedAttr rebound = evaluator.getReboundAttribute(typeAttr);
    if (!rebound)
      return std::nullopt;
    fieldTypes.push_back(ParamType::get(rebound));
  }
  return fieldTypes;
}

FailureOr<TypedAttr> StructFieldTypesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  if (auto typeParam = dyn_cast<TypeParamAttr>(getTypeValue())) {
    if (auto structType =
            sugarDynCast<KGEN::StructType>(typeParam.getTypeValue())) {
      auto elementTypes = structType.getElementTypes();
      if (!elementTypes)
        return failure();
      SmallVector<TypedAttr> resultAttrs;
      resultAttrs.reserve(elementTypes->size());
      Type resultElemType = getType().getElementType();
      for (Type fieldType : *elementTypes)
        resultAttrs.push_back(TypeParamAttr::get(fieldType, resultElemType));
      return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
    }
  }

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_types requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;

  // If concrete instance is available, use its already-substituted field types.
  if (resolved.instance) {
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    SmallVector<TypedAttr> resultAttrs;
    for (StructDefFieldAttr field : structType.getFields())
      resultAttrs.push_back(field.getTypeValue());
    return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Otherwise, use generator types and rebind with param values.
  SmallVector<TypedAttr> fieldTypes;
  resolved.decl.getFieldTypes(fieldTypes, getType().getElementType());

  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        SmallVector<TypedAttr> resultAttrs;
        for (TypedAttr fieldType : fieldTypes) {
          TypedAttr rebound = evaluator.getReboundAttribute(fieldType);
          if (!rebound) {
            result = TypedAttr();
            return;
          }
          resultAttrs.push_back(rebound);
        }
        result = cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
      });
  return result;
}

FailureOr<TypedAttr> StructFieldNamesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_names requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  SmallVector<StringAttr> fieldNames;
  resolved.decl.getFieldNames(fieldNames);

  SmallVector<TypedAttr> resultAttrs;
  MLIRContext *ctx = getContext();
  for (StringAttr name : fieldNames)
    resultAttrs.push_back(
        StringAttr::get(name.getValue(), StringType::get(ctx)));

  return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
}

//===----------------------------------------------------------------------===//
// Decorator Reflection Attrs
//===----------------------------------------------------------------------===//

namespace {
/// A struct declaration plus the decorators applied to it, or to one of its
/// fields.
struct DecoratorSite {
  /// The resolved struct. `decl` is null when async concretization was
  /// triggered and the caller should retry later.
  ResolvedStructHandle resolved;
  /// The stored decorator values, in application order.
  SmallVector<TypedAttr> decorators;
  /// The decorator struct each value came from, parallel to `decorators`.
  /// Individual entries may be null when identity is unrecoverable.
  SmallVector<SymbolRefAttr> typeNames;
  /// The type of the decorated declaration -- the struct's own type for a
  /// struct decorator, the field's type for a field decorator. This is the
  /// value the compiler bound the (at most one) parameter of every decorator
  /// here to, so it is the second half of decorator identity: base-struct
  /// symbol plus this reconstructs the full parameterization. Null when it
  /// could not be recovered, in which case the parameterization is not
  /// checked and identity falls back to the symbol alone.
  ///
  /// A field's type may reference the site struct's own parameters, in which
  /// case `siteTypeNeedsRebinding` is set and it is rebound through the same
  /// evaluator the stored decorator values are. A struct's own type comes
  /// straight from the attribute's operand and is already in the caller's
  /// world, so it must not be.
  TypedAttr siteType;
  bool siteTypeNeedsRebinding = false;
};

/// A decorator query decoded into the two halves of decorator identity: the
/// base struct's symbol, and the parameterization the caller spelled.
struct DecoratorQuery {
  /// The decorator struct's identity symbol, in the namespace the site
  /// records identity in. Null when the operand does not (yet) name a struct.
  SymbolRefAttr symbol;
  /// The parameter values bound in the query (`tagged[Int]` -> `[Int]`).
  /// Empty for an unparameterized decorator struct -- but only meaningful
  /// when `parameterizationResolved` is set.
  ArrayRef<TypedAttr> paramValues;
  /// Whether `paramValues` could be determined at all. When false the query
  /// matches nothing, rather than matching on the symbol alone: skipping the
  /// check would be a *wrong-answer* path, handing back a stored
  /// `tagged[String]` in a list typed `tagged[Int]`.
  bool parameterizationResolved = true;

  explicit operator bool() const { return static_cast<bool>(symbol); }
};

/// Reduce a parameter value that denotes a type to the type it denotes, so
/// that two spellings differing only in metatype (`!kgen.type` versus a trait
/// metatype, with or without a wrapper) still compare equal. Returns null when
/// the value does not denote a type.
Type asTypeValue(TypedAttr attr) {
  if (auto typeParam = sugarDynCastIfPresent<TypeParamAttr>(attr))
    return typeParam.getTypeValue();
  return {};
}
} // namespace

/// Decode the `decoratorType` operand of a `*_decorator_of` attribute into the
/// identity symbol to compare against `StructDeclInterface::
/// get{Struct,Field}DecoratorTypeNames`, plus the parameterization the caller
/// spelled. Returns a falsy query if the operand does not (yet) name a struct
/// declaration.
///
/// Identity has to be compared on the decorator's struct rather than on the
/// stored value's type, because lowering anonymizes that type: a fieldless
/// decorator becomes `#kgen.struct<> : !kgen.struct<() memoryOnly>`, which is
/// byte-identical for two different decorator structs.
///
/// The symbol is not quite the whole of identity, because a decorator struct
/// may declare one parameter. It is *almost* the whole of it: that parameter
/// is bound by the compiler to the type of the decorated declaration, never
/// to anything the user chose, so base-struct symbol plus the site
/// reconstructs the parameterization exactly. `paramValues` is therefore
/// checked against the site's own type (see `packDecorators`) rather than
/// against anything recorded per stored value.
///
/// The symbol namespace depends on which op `site` is, because the two
/// implementations of the interface record identity differently:
///
///  * `lit.struct.decl` (parser) reports the LIT struct symbol carried by the
///    decorator value's own type, e.g. `@mod::@serde`.
///  * `kgen.struct.generator` (KGEN symbol table, LowerLIT, and both
///    elaborators) reports the flattened generator symbol LowerLIT recorded,
///    e.g. `@"mod::serde"`.
///
/// Resolving the operand through the *same* context is what makes the second
/// case agree: whichever form the operand takes -- a LIT struct type value
/// under LowerLIT, a `genref`, or the `instref` an elaborator produces for a
/// concretized type -- `resolveStructOp` lands it on the generator whose
/// `sym_name` the recorded identity is.
static DecoratorQuery decodeDecoratorQuery(ParameterEvaluationContext &context,
                                           StructDeclInterface site,
                                           TypedAttr decoratorType) {
  if (!isa<StructGeneratorOp>(site.getOperation())) {
    // Parser: the operand is a struct type value of exactly the shape the
    // stored identity was read from, so the symbol needs no resolution.
    auto typeParam = sugarDynCastIfPresent<TypeParamAttr>(decoratorType);
    if (!typeParam)
      return {};
    auto named =
        sugarDynCastIfPresent<StructTypeInterface>(typeParam.getTypeValue());
    if (!named)
      return {};
    // The parameterization does need resolving, and through the same route
    // the *site* is resolved by (`resolveDecoratorSite`), so the two are
    // spelled in the same world when compared. A failure makes the query
    // match nothing rather than fall back to matching on the symbol alone:
    // the latter would return a stored `tagged[String]` in a list typed
    // `tagged[Int]`.
    //
    // Failing closed costs nothing in practice. `LIT::StructType` is the only
    // implementer of `StructTypeInterface`, so reaching here at all means the
    // operand *is* a LIT struct type value; the resolution that follows can
    // then only fail if resolving the decorator struct's own body fails --
    // which is separately diagnosed, so there is no silently-lost match.
    FailureOr<ResolvedStructHandle> queryOr =
        context.resolveStructOp(decoratorType, /*acceptAsync=*/false);
    if (failed(queryOr) || !queryOr->decl)
      return {named.getSymbolRef(), {}, /*parameterizationResolved=*/false};
    return {named.getSymbolRef(), queryOr->paramValues};
  }

  FailureOr<ResolvedStructHandle> queryOr =
      context.resolveStructOp(decoratorType, /*acceptAsync=*/false);
  if (failed(queryOr) || !queryOr->decl)
    return {};
  auto gen = dyn_cast<StructGeneratorOp>(queryOr->decl.getOperation());
  if (!gen)
    return {};
  return {SymbolRefAttr::get(gen.getSymNameAttr()), queryOr->paramValues};
}

/// Resolve the struct named by `typeValue` and collect the decorators applied
/// to the struct itself, or -- when `fieldIndex` is non-null -- to the field
/// it indexes. `attrName` names the attribute in diagnostics.
///
/// Returns `failure()` when `typeValue` is not a struct type (a materialization
/// error is emitted), when `fieldIndex` is out of range (likewise), or when
/// `fieldIndex` has not folded to a constant yet (silently, to be retried).
/// A returned site with a null `resolved.decl` means async concretization was
/// triggered and the caller should return a null result to be retried.
static FailureOr<DecoratorSite>
resolveDecoratorSite(ParameterEvaluationContext &context, StringRef attrName,
                     TypedAttr typeValue, TypedAttr fieldIndex) {
  // Wait until the index has folded to a constant (it may be a nested
  // struct_field_index_by_name or a Mojo Int parameter expression).
  IntegerAttr fieldIndexAttr;
  if (fieldIndex) {
    fieldIndexAttr = dyn_cast<IntegerAttr>(fieldIndex);
    if (!fieldIndexAttr)
      return failure();
  }

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(typeValue, /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(Twine(attrName) +
                                     " requires a struct type");
    return failure();
  }

  DecoratorSite site;
  site.resolved = *resolvedOr;
  if (!site.resolved.decl)
    return site; // Async: not ready yet.

  if (!fieldIndexAttr) {
    site.resolved.decl.getStructDecorators(site.decorators);
    site.resolved.decl.getStructDecoratorTypeNames(site.typeNames);
    // A struct decorator's site type is the struct's own type -- which is
    // exactly the operand that named it. Already in the caller's world, so
    // unlike the field case it must *not* be rebound through the struct's
    // own parameter scope.
    site.siteType = typeValue;
    return site;
  }

  SmallVector<StringAttr> fieldNames;
  site.resolved.decl.getFieldNames(fieldNames);
  int64_t index = fieldIndexAttr.getInt();
  if (index < 0 || index >= int64_t(fieldNames.size())) {
    context.emitMaterializationError(
        Twine(attrName) + ": field index " + Twine(index) +
        " out of range for struct with " + Twine(fieldNames.size()) +
        " fields");
    return failure();
  }
  site.resolved.decl.getFieldDecorators(size_t(index), site.decorators);
  site.resolved.decl.getFieldDecoratorTypeNames(size_t(index), site.typeNames);
  // A field decorator's site type is the field's type. It may reference the
  // struct's parameters, so it is rebound alongside the decorator values --
  // `siteTypeNeedsRebinding` records that.
  SmallVector<TypedAttr> fieldTypes;
  site.resolved.decl.getFieldTypes(
      fieldTypes, TypeType::get(site.resolved.decl.getContext()));
  if (size_t(index) < fieldTypes.size()) {
    site.siteType = fieldTypes[size_t(index)];
    site.siteTypeNeedsRebinding = true;
  }
  return site;
}

/// Shared tail for the `*_decorator_of` attributes: rebind each stored
/// decorator (a value that may refer to the struct's own parameters) through
/// the struct's parameter values, keep the ones the `wanted` query identifies,
/// and package the result as a param_list of `resultType`.
///
/// Rebinding still happens for every stored decorator, including the ones
/// `wanted` filters out, so that a decorator whose payload depends on the
/// struct's parameters is specialized before anything else looks at it.
///
/// Matching is on the two halves of decorator identity. The base struct's
/// symbol is compared per entry, against the parallel identity array. The
/// parameterization is compared once, for the query as a whole, against the
/// site's own type: a decorator struct's single parameter is always bound by
/// the compiler to the type of the declaration it is attached to, so every
/// decorator recorded at this site carries the same binding and the query
/// either agrees with it or matches nothing here. That is what keeps
/// `decorator_of[tagged[String]]` from answering on an `Int` field, and it
/// needs no per-value record beyond the symbol.
static FailureOr<TypedAttr>
packDecorators(ParameterEvaluationContext &context, DecoratorSite &site,
               ParamListType resultType, const DecoratorQuery &wanted) {
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      site.resolved.decl.getInputParams(), site.resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        // The query is parameterized iff the decorator struct declares a
        // parameter; when it does, it must be the site's type. A site type
        // that could not be recovered at all matches nothing, which is the
        // same fail-closed answer `decoratorValueTypeAttr` gives to that
        // condition -- see the note there.
        std::optional<bool> parameterizationMatches;
        if (!wanted.parameterizationResolved) {
          parameterizationMatches = false;
        } else if (!wanted.paramValues.empty()) {
          TypedAttr siteType = site.siteType;
          if (siteType && site.siteTypeNeedsRebinding)
            siteType = evaluator.getReboundAttribute(siteType);
          Type siteTypeValue = asTypeValue(siteType);
          Type queryTypeValue = asTypeValue(wanted.paramValues.front());
          // Canonical equality, not pointer equality: a field's declared type
          // keeps its sugar (`!kgen.param<:meta<!Int> #alias_Int>`) while the
          // query's parameterization arrives desugared (`!Int`).
          parameterizationMatches = wanted.paramValues.size() == 1 &&
                                    siteTypeValue && queryTypeValue &&
                                    isEqualCanon(queryTypeValue, siteTypeValue);
        }

        SmallVector<TypedAttr> resultAttrs;
        for (auto [i, decorator] : llvm::enumerate(site.decorators)) {
          TypedAttr rebound = evaluator.getReboundAttribute(decorator);
          if (!rebound) {
            result = TypedAttr();
            return;
          }
          SymbolRefAttr typeName =
              i < site.typeNames.size() ? site.typeNames[i] : SymbolRefAttr();
          if (typeName != wanted.symbol)
            continue;
          if (parameterizationMatches && !*parameterizationMatches)
            continue;
          resultAttrs.push_back(rebound);
        }
        result = cast<TypedAttr>(ParamListAttr::get(resultAttrs, resultType));
      });
  return result;
}

/// Find the `kgen.struct.generator` a recorded decorator-identity symbol names,
/// searching outward from `site` through enclosing symbol tables.
///
/// `SymbolTable::lookupNearestSymbolFrom` is not enough: a struct generator is
/// itself a symbol table, so that helper searches only *inside* the decorated
/// struct and stops. Decorator generators are siblings at module scope.
///
/// Returns null when the symbol cannot be found. Callers must treat that as
/// *unrecoverable identity*, not as "no parameters": a decorator that does
/// declare one would otherwise get an arity-mismatched `genref`, which is the
/// shape that asserts in name mangling.
///
/// That is a deliberate trade, not a free one. A miss on a *parameterless*
/// decorator used to be harmless -- the reference it produced bound nothing
/// either way, so identity survived -- and now costs identity there too. Both
/// readings cannot be had at once without knowing the parameter count, which
/// is exactly what a miss denies, so this fails closed: a wrong (empty)
/// answer beats an assertion. A miss should be unreachable regardless; see
/// the note in `decoratorValueTypeAttr`.
static StructGeneratorOp lookupStructGeneratorFrom(StructDeclInterface site,
                                                   SymbolRefAttr symbol) {
  for (Operation *op = site.getOperation(); op; op = op->getParentOp())
    if (op->hasTrait<OpTrait::SymbolTable>())
      if (auto gen =
              dyn_cast_if_present<StructGeneratorOp>(
                  SymbolTable::lookupSymbolIn(op, symbol)))
        return gen;
  return nullptr;
}

/// Build the type value that `*_decorator_types` reports for one stored
/// decorator, given the rebound value's own type and the identity symbol
/// recorded alongside it.
///
/// This is the emitting counterpart of `decodeDecoratorQuery`, and it splits
/// on the same thing -- which op the site is -- for the same reason:
///
///  * `lit.struct.decl` (parser): the value's type is still
///    `!lit.struct<@mod::@serde>`, which names the decorator struct, so it is
///    reported as-is.
///  * `kgen.struct.generator` (post-LowerLIT, both elaborators): the value's
///    type has been anonymized to a bare `!kgen.struct<...>`, identical for
///    any two fieldless decorators. Reporting it would make
///    `decorator_types()[0] == tag` false for the decorator that is actually
///    there and true for a different one, so the type-domain half is rebuilt
///    from the recorded generator symbol instead; the anonymized type stays
///    on as the value-domain half, which is what it correctly describes.
///
/// A decorator struct may declare one parameter, and it is always bound to
/// `siteType` -- the type of the decorated declaration. The rebuilt reference
/// has to bind it too: an arity-mismatched `genref` is not merely a wrong
/// answer, it asserts in name mangling, and a caller feeding
/// `decorator_types()` back into `decorator_of` (see `packDecorators`) must
/// get the same parameterization back that the query has to spell.
///
/// Falls back to the value's own type when identity was not recoverable
/// (a null `typeName`) -- the same "unknown identity" encoding that makes a
/// stored decorator match no `*_decorator_of` query.
///
/// Returns null when the rebuilt reference is not resolvable yet, which the
/// caller treats the same way it treats a value that has not rebound.
static TypedAttr decoratorValueTypeAttr(ParameterEvaluator &evaluator,
                                        StructDeclInterface site,
                                        SymbolRefAttr typeName, Type valueType,
                                        Type elementType, TypedAttr siteType) {
  if (!typeName || !isa<StructGeneratorOp>(site.getOperation()))
    return TypeParamAttr::get(valueType, elementType);

  MLIRContext *ctx = site.getContext();
  // The decorator generator has to be in hand before a reference can be
  // rebuilt: without it the parameter count is unknown, and guessing "none"
  // would rebuild the arity-mismatched `genref` that asserts in mangling.
  // A miss is therefore the unrecoverable-identity encoding, exactly as a
  // null `typeName` is -- such a value matches no `*_decorator_of` query.
  StructGeneratorOp decoratorGen = lookupStructGeneratorFrom(site, typeName);
  if (!decoratorGen)
    return TypeParamAttr::get(valueType, elementType);

  SmallVector<TypedAttr> paramValues;
  ArrayRef<ParamDeclAttr> params = decoratorGen.getInputParams();
  if (!params.empty()) {
    // A missing site type is the *unrecoverable identity* encoding here, the
    // same as a missing generator above -- not a null "not folded yet, retry"
    // result, which for a condition that never becomes true would spin rather
    // than diagnose. `packDecorators` answers the identical condition the
    // identical way (no match), and the two must not disagree: only a
    // `getFieldTypes`/`getFieldNames` length mismatch can produce it, which
    // is impossible by construction, so what matters is that neither
    // interpretation can hang.
    if (!siteType)
      return TypeParamAttr::get(valueType, elementType);
    paramValues.push_back(
        ParamOperatorAttr::getRebind(siteType, params.front().getType()));
  }
  auto genRef =
      TypeGeneratorRefAttr::get(typeName, paramValues, TypeType::get(ctx));
  // Run the reference back through the evaluator so it is concretized just as
  // one written in source is: an elaborator rewrites it to the `instref` that
  // a caller's own spelling of the decorator type has become by the time the
  // two are compared, and comparing a `genref` against an `instref` would
  // decide neither equal nor unequal and leave the comparison unfolded.
  return evaluator.getReboundAttribute(
      TypeParamAttr::get(TypeValueType::get(genRef), valueType, elementType));
}

/// Shared tail for the `*_decorator_types` attributes: rebind each stored
/// decorator as `packDecorators` does, but yield the *type* of each value
/// rather than the value, converted to `resultType`'s element metatype.
static FailureOr<TypedAttr>
packDecoratorValueTypes(ParameterEvaluationContext &context,
                        DecoratorSite &site, ParamListType resultType) {
  Type elementType = resultType.getElementType();
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      site.resolved.decl.getInputParams(), site.resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        TypedAttr reboundSiteType = site.siteType;
        if (reboundSiteType && site.siteTypeNeedsRebinding)
          reboundSiteType = evaluator.getReboundAttribute(reboundSiteType);
        SmallVector<TypedAttr> resultAttrs;
        resultAttrs.reserve(site.decorators.size());
        for (auto [i, decorator] : llvm::enumerate(site.decorators)) {
          TypedAttr rebound = evaluator.getReboundAttribute(decorator);
          if (!rebound) {
            result = TypedAttr();
            return;
          }
          SymbolRefAttr typeName =
              i < site.typeNames.size() ? site.typeNames[i] : SymbolRefAttr();
          TypedAttr typeValue = decoratorValueTypeAttr(
              evaluator, site.resolved.decl, typeName, rebound.getType(),
              elementType, reboundSiteType);
          if (!typeValue) {
            result = TypedAttr();
            return;
          }
          resultAttrs.push_back(typeValue);
        }
        result = cast<TypedAttr>(ParamListAttr::get(resultAttrs, resultType));
      });
  return result;
}

FailureOr<TypedAttr> StructDecoratorTypesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FailureOr<DecoratorSite> site =
      resolveDecoratorSite(context, "struct_decorator_types", getTypeValue(),
                           /*fieldIndex=*/TypedAttr());
  if (failed(site))
    return failure();
  if (!site->resolved.decl)
    return TypedAttr(); // Async: not ready yet.
  return packDecoratorValueTypes(context, *site, getType());
}

/// Defense in depth for the "at most one" invariant the singular
/// `*_decorator_of` attributes promise. `packDecorators` filters by identity
/// but does not itself cap the result at one entry -- these attributes get
/// away with a 0-or-1 result only because callers currently guarantee it
/// from outside this file: the parser's `dedupeDecorators` refuses a second
/// non-repeatable decorator sharing a base struct, and the stdlib's
/// `decorator_of[D]()` separately refuses to even ask this question for a
/// `D` conforming to `RepeatableDecorator` (see `reflect.mojo`). Both of
/// those are bypassable from here -- a hand-written `__mlir_attr` spelling
/// can name any decorator type, repeatable or not -- so this is the layer
/// that actually *owns* the invariant, and it is enforced here too rather
/// than solely trusted from callers.
static FailureOr<TypedAttr> requireAtMostOneMatch(
    ParameterEvaluationContext &context, StringRef attrName,
    FailureOr<TypedAttr> result) {
  if (failed(result) || !*result)
    return result;
  auto paramList = sugarDynCast<ParamListAttr>(*result);
  if (paramList && paramList.getValues().size() > 1) {
    context.emitMaterializationError(
        Twine(attrName) + " found " +
        Twine(paramList.getValues().size()) +
        " matching decorators, but at most one is expected here; a "
        "decorator conforming to RepeatableDecorator must be queried with "
        "the plural form instead");
    return failure();
  }
  return result;
}

FailureOr<TypedAttr> StructDecoratorOfAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // Resolve the site first: it owns the "not a struct type" diagnostic, and
  // for the field variants the out-of-range index one, which would otherwise
  // be lost whenever the queried decorator type is also unresolvable.
  FailureOr<DecoratorSite> site =
      resolveDecoratorSite(context, "struct_decorator_of", getTypeValue(),
                           /*fieldIndex=*/TypedAttr());
  if (failed(site))
    return failure();
  if (!site->resolved.decl)
    return TypedAttr(); // Async: not ready yet.

  DecoratorQuery wanted =
      decodeDecoratorQuery(context, site->resolved.decl, getDecoratorType());
  if (!wanted) {
    context.emitMaterializationError(
        "struct_decorator_of requires a decorator struct type");
    return failure();
  }
  return requireAtMostOneMatch(
      context, "struct_decorator_of",
      packDecorators(context, *site, getType(), wanted));
}

FailureOr<TypedAttr> StructDecoratorsOfAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // Site first, for the same reason `StructDecoratorOfAttr` does: it owns the
  // "not a struct type" diagnostic.
  FailureOr<DecoratorSite> site =
      resolveDecoratorSite(context, "struct_decorators_of", getTypeValue(),
                           /*fieldIndex=*/TypedAttr());
  if (failed(site))
    return failure();
  if (!site->resolved.decl)
    return TypedAttr(); // Async: not ready yet.

  DecoratorQuery wanted =
      decodeDecoratorQuery(context, site->resolved.decl, getDecoratorType());
  if (!wanted) {
    context.emitMaterializationError(
        "struct_decorators_of requires a decorator struct type");
    return failure();
  }
  // `packDecorators` already collects every matching entry rather than
  // stopping at the first -- the non-repeatable `*_decorator_of` attributes
  // get away with 0-or-1 only because the parser guarantees at most one
  // match exists for them, not because this helper enforces it. So the
  // plural form needs no changes here, only its own identity (mnemonic and
  // diagnostic text).
  return packDecorators(context, *site, getType(), wanted);
}

FailureOr<TypedAttr> StructFieldDecoratorTypesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FailureOr<DecoratorSite> site = resolveDecoratorSite(
      context, "struct_field_decorator_types", getTypeValue(), getFieldIndex());
  if (failed(site))
    return failure();
  if (!site->resolved.decl)
    return TypedAttr(); // Async: not ready yet.
  return packDecoratorValueTypes(context, *site, getType());
}

FailureOr<TypedAttr> StructFieldDecoratorOfAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // Site first, so the out-of-range field index diagnostic is not lost when
  // the queried decorator type is also unresolvable. See above.
  FailureOr<DecoratorSite> site = resolveDecoratorSite(
      context, "struct_field_decorator_of", getTypeValue(), getFieldIndex());
  if (failed(site))
    return failure();
  if (!site->resolved.decl)
    return TypedAttr(); // Async: not ready yet.

  DecoratorQuery wanted =
      decodeDecoratorQuery(context, site->resolved.decl, getDecoratorType());
  if (!wanted) {
    context.emitMaterializationError(
        "struct_field_decorator_of requires a decorator struct type");
    return failure();
  }
  // See `requireAtMostOneMatch`: defense in depth for the "at most one"
  // invariant, enforced here rather than solely trusted from callers.
  return requireAtMostOneMatch(
      context, "struct_field_decorator_of",
      packDecorators(context, *site, getType(), wanted));
}

FailureOr<TypedAttr> StructFieldDecoratorsOfAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // Site first, so the out-of-range field index diagnostic is not lost when
  // the queried decorator type is also unresolvable. See above.
  FailureOr<DecoratorSite> site = resolveDecoratorSite(
      context, "struct_field_decorators_of", getTypeValue(), getFieldIndex());
  if (failed(site))
    return failure();
  if (!site->resolved.decl)
    return TypedAttr(); // Async: not ready yet.

  DecoratorQuery wanted =
      decodeDecoratorQuery(context, site->resolved.decl, getDecoratorType());
  if (!wanted) {
    context.emitMaterializationError(
        "struct_field_decorators_of requires a decorator struct type");
    return failure();
  }
  // See `StructDecoratorsOfAttr::evaluateWithContext`: `packDecorators`
  // already returns every match, so the plural field form reuses it as-is.
  return packDecorators(context, *site, getType(), wanted);
}

//===----------------------------------------------------------------------===//
// Function Reflection Attrs
//===----------------------------------------------------------------------===//

namespace {
/// Resolve a function-valued `TypedAttr` to its defining op via the
/// evaluation context. Returns null if the value is not a direct function
/// reference (`#kgen.symbol.constant<@...>`).
///
/// The returned `FuncInterface` op is `lit.fn` when resolved through the
/// parser or LIT symbol-table contexts, or `kgen.generator` when resolved
/// through the KGEN symbol-table or IR evaluator contexts. Reflection
/// attrs are evaluated during parsing or during elaboration, so the
/// post-elaboration `kgen.func` form is never the resolution target. Both
/// reachable ops also implement `DeclInterface`, so callers needing the
/// param list cast accordingly.
FuncInterface resolveFuncDecl(TypedAttr funcValue,
                              ParameterEvaluationContext &context) {
  // Mojo function values reach reflection as `#kgen.symbol.constant<@func>`.
  // Closure literals are not yet supported.
  auto symbol = dyn_cast<SymbolConstantAttr>(funcValue);
  if (!symbol)
    return nullptr;
  return context.resolveFunctionDecl(symbol.getSymbol());
}
} // namespace

FailureOr<TypedAttr> GetFunctionParameterCountAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FuncInterface func = resolveFuncDecl(getFunc(), context);
  if (!func) {
    context.emitMaterializationError(
        "get_function_parameter_count requires a concrete function value");
    return failure();
  }
  // Prefer the source-declared parameter list snapshot on `kgen.generator`
  // when available so reflection counts remain stable across transforms that
  // rewrite the live `inputParams` (e.g. `RemoveUnusedParams`). Falls back to
  // the live input params for `lit.fn` (pre-LowerLIT reflection) and for
  // generators that don't carry a snapshot.
  size_t count;
  if (auto gen = dyn_cast<GeneratorOp>(func.getOperation())) {
    if (PogListAttr snapshot = gen.getSourceParamListAttr()) {
      count = snapshot.size();
    } else {
      count = gen.getInputParams().size();
    }
  } else {
    count = cast<DeclInterface>(func.getOperation()).getInputParams().size();
  }
  return cast<TypedAttr>(IntegerAttr::get(IndexType::get(getContext()), count));
}

FailureOr<TypedAttr> GetFunctionParameterNamesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FuncInterface func = resolveFuncDecl(getFunc(), context);
  if (!func) {
    context.emitMaterializationError(
        "get_function_parameter_names requires a concrete function value");
    return failure();
  }
  MLIRContext *ctx = getContext();
  SmallVector<TypedAttr> resultAttrs;

  auto appendName = [&](StringAttr name) {
    resultAttrs.push_back(
        StringAttr::get(name.getValue(), StringType::get(ctx)));
  };

  // Prefer the source-declared parameter list snapshot on `kgen.generator`
  // when available; see the comment in `GetFunctionParameterCountAttr`.
  if (auto gen = dyn_cast<GeneratorOp>(func.getOperation())) {
    if (PogListAttr snapshot = gen.getSourceParamListAttr()) {
      resultAttrs.reserve(snapshot.size());
      for (PogMetadataAttr pog : snapshot.getPogs())
        appendName(pog.getName());
      return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
    }
  }

  ArrayRef<ParamDeclAttr> params =
      cast<DeclInterface>(func.getOperation()).getInputParams();
  resultAttrs.reserve(params.size());
  for (ParamDeclAttr param : params)
    appendName(param.getName());
  return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
}

FailureOr<TypedAttr> GetFunctionIsRaisingAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FuncInterface func = resolveFuncDecl(getFunc(), context);
  if (!func) {
    context.emitMaterializationError(
        "get_function_is_raising requires a concrete function value");
    return failure();
  }
  return cast<TypedAttr>(BoolAttr::get(getContext(), func.isThrows()));
}

FailureOr<TypedAttr> StructFieldIndexByNameAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto fieldNameAttr = dyn_cast<StringAttr>(getFieldName());
  if (!fieldNameAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_index_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  auto index = resolved.decl.findFieldIndex(fieldNameAttr.getValue());
  if (!index) {
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldNameAttr.getValue() + "'");
    return failure();
  }
  return cast<TypedAttr>(Builder(getType().getContext()).getIndexAttr(*index));
}

FailureOr<TypedAttr> StructFieldTypeByNameAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto fieldNameAttr = dyn_cast<StringAttr>(getFieldName());
  if (!fieldNameAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_type_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  StringRef fieldName = fieldNameAttr.getValue();

  // If concrete instance is available, search its fields directly.
  if (resolved.instance) {
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    for (StructDefFieldAttr field : structType.getFields())
      if (field.getName().getValue() == fieldName)
        return field.getTypeValue();
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Otherwise, use generator's field type and rebind.
  TypedAttr fieldType = resolved.decl.getFieldType(fieldName, getType());
  if (!fieldType) {
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }

  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                        [&](ParameterEvaluator &evaluator) {
                          result = evaluator.getReboundAttribute(fieldType);
                        });
  return result;
}

FailureOr<TypedAttr> StructFieldOffsetByIndexAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // Return failure() without an error if parameters aren't resolved to
  // constants yet. The evaluation framework will retry later when more
  // information is available.
  auto fieldIndexAttr = dyn_cast<IntegerAttr>(getFieldIndex());
  if (!fieldIndexAttr)
    return failure();

  auto targetAttr = sugarDynCast<TargetParamAttr>(getTarget());
  if (!targetAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_offset_by_index requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  int64_t fieldIndex = fieldIndexAttr.getInt();
  TargetInfoAttr target = targetAttr.getTarget();
  MLIRContext *ctx = getType().getContext();

  auto emitError = [&context](const Twine &msg) {
    context.emitMaterializationError(msg);
  };

  // If concrete instance is available, use its field types directly.
  if (resolved.instance) {
    assert(resolved.decl && "instance requires valid decl");
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    SmallVector<Type> fieldTypes = getFieldTypesFromStruct(structType);

    FailureOr<int64_t> offsetOr =
        computeStructFieldOffset(fieldTypes, fieldIndex, target, emitError);
    if (failed(offsetOr))
      return failure();
    return cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Otherwise, use generator's field types with rebinding.
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        std::optional<SmallVector<Type>> fieldTypesOpt =
            rebindFieldTypes(resolved.decl, evaluator);
        if (!fieldTypesOpt) {
          result = TypedAttr();
          return;
        }

        FailureOr<int64_t> offsetOr = computeStructFieldOffset(
            *fieldTypesOpt, fieldIndex, target, emitError);
        if (failed(offsetOr))
          return;
        result = cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
      });
  return result;
}

FailureOr<TypedAttr> StructFieldOffsetByNameAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // Return failure() without an error if parameters aren't resolved to
  // constants yet.
  auto fieldNameAttr = dyn_cast<StringAttr>(getFieldName());
  if (!fieldNameAttr)
    return failure();

  auto targetAttr = sugarDynCast<TargetParamAttr>(getTarget());
  if (!targetAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_offset_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  StringRef fieldName = fieldNameAttr.getValue();
  TargetInfoAttr target = targetAttr.getTarget();
  MLIRContext *ctx = getType().getContext();

  auto emitError = [&context](const Twine &msg) {
    context.emitMaterializationError(msg);
  };

  // Helper to find field index by name and emit error if not found.
  auto findFieldIndexOrError =
      [&](auto fields, StringRef structName) -> std::optional<int64_t> {
    int64_t idx = 0;
    for (auto field : fields) {
      if (field.getName().getValue() == fieldName)
        return idx;
      ++idx;
    }
    context.emitMaterializationError(
        "struct '" + structName + "' has no field named '" + fieldName + "'");
    return std::nullopt;
  };

  // If concrete instance is available, use its fields directly.
  if (resolved.instance) {
    assert(resolved.decl && "instance requires valid decl");
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    auto fields = structType.getFields();
    StringRef structName =
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue();

    std::optional<int64_t> fieldIndexOpt =
        findFieldIndexOrError(fields, structName);
    if (!fieldIndexOpt)
      return failure();

    SmallVector<Type> fieldTypes = getFieldTypesFromStruct(structType);
    FailureOr<int64_t> offsetOr =
        computeStructFieldOffset(fieldTypes, *fieldIndexOpt, target, emitError);
    if (failed(offsetOr))
      return failure();
    return cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Find field index using the decl.
  std::optional<uint64_t> fieldIndexOpt =
      resolved.decl.findFieldIndex(fieldName);
  if (!fieldIndexOpt) {
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }
  int64_t fieldIndex = static_cast<int64_t>(*fieldIndexOpt);

  // Use generator's field types with rebinding.
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        std::optional<SmallVector<Type>> fieldTypesOpt =
            rebindFieldTypes(resolved.decl, evaluator);
        if (!fieldTypesOpt) {
          result = TypedAttr();
          return;
        }

        FailureOr<int64_t> offsetOr = computeStructFieldOffset(
            *fieldTypesOpt, fieldIndex, target, emitError);
        if (failed(offsetOr))
          return;
        result = cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
      });
  return result;
}
