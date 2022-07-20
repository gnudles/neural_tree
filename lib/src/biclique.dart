import 'package:neural_tree/src/activation_function.dart';
import 'package:neural_tree/src/component.dart';
import 'package:neural_tree/src/linalg.dart';
import 'package:neural_tree/src/node.dart';

class BicliqueBackProducts extends LayerBackwardProducts {
  BicliqueDelta get delta => abstractDelta as BicliqueDelta;
  const BicliqueBackProducts(
      FVector forwardError, FVector propagatedError, Delta delta)
      : super(forwardError, propagatedError, delta);
}

class BicliqueDelta implements Delta {
  int get outWidth => _weight.nRows;
  int get inWidth => _weight.nColumns;
  FLeftMatrix get weight => _weight;
  FVector get bias => _bias;
  final FLeftMatrix _weight;
  final FVector _bias;
  BicliqueDelta(this._weight, this._bias);
  factory BicliqueDelta.fromJson(Map<String, dynamic> map) {
    return BicliqueDelta(
        FLeftMatrix.fromJson(map['w']), FVector.fromJson(map['b']));
  }

  @override
  void add(Delta other) {
    if (other is BicliqueDelta) {
      this._weight.add(other._weight);
      this._bias.add(other._bias);
    }
  }

  @override
  void scale(double factor) {
    this._weight.scale(factor);
    this._bias.scale(factor);
  }

  @override
  Map<String, dynamic> toJson() {
    return {'type': 'biclique', 'w': _weight.toJson(), 'b': _bias.toJson()};
  }

  @override
  void clamp(double maxVal) {
    this._weight.clamp(-maxVal, maxVal);
    this._bias.clamp(-maxVal, maxVal);
  }
}

/// Fully Connected Layer.
///
class BicliqueComponent extends Component {
  final ActivationFunction _activationFunc;
  final FLeftMatrix _weight;
  final FVector _bias;
  BicliqueComponent(int inWidth, int outWidth, this._weight, this._bias,
      ActivationFunctionType activationFuncType)
      : _activationFunc = mapActivationFunction[activationFuncType]!,
        super(inWidth, outWidth);
  factory BicliqueComponent.random(
      int inWidth,
      int outWidth,
      ActivationFunctionType activationFuncType,
      double Function() randomBias,
      double Function() randomWeight) {
    FLeftMatrix weight =
        FLeftMatrix.generate(inWidth, outWidth, (c, r) => randomWeight());
    FVector bias = FVector.generate(outWidth, (i) => randomBias());
    return BicliqueComponent(
        inWidth, outWidth, weight, bias, activationFuncType);
  }
  factory BicliqueComponent.fromJson(Map<String, dynamic> spec) {
    FLeftMatrix weight = FLeftMatrix.fromJson(spec["weight"]);
    FVector bias = FVector.fromJson(spec["bias"]);
    ActivationFunctionType activationFuncType =
        activationTypeFromString[spec["activation"]]!;
    return BicliqueComponent(
        weight.nColumns, weight.nRows, weight, bias, activationFuncType);
  }

  Map<String, dynamic> toJson() {
    return {
      "type": "biclique",
      "weight": _weight.toJson(),
      "bias": _bias.toJson(),
      "activation": _activationFunc.type.toString()
    };
  }

  @override
  FVector feedForward(FVector input) {
    return ((_weight.multiplyVector(input)) + _bias)
      ..apply(_activationFunc.func, _activationFunc.funcSIMD);
  }

  @override
  ComponentForwardProducts produce(FVector input) {
    FVector intermediateVector = (_weight.multiplyVector(input)) + _bias;
    return LayerFwdProducts(
        input,
        intermediateVector.applied(
            _activationFunc.func, _activationFunc.funcSIMD),
        intermediateVector.applied(
            _activationFunc.derivative, _activationFunc.derivativeSIMD));
  }

  @override
  ComponentBackwardProducts backPropagate(
      ForwardProducts fwdProducts, FVector err) {
    if (!(fwdProducts is LayerFwdProducts)) throw ArgumentError();
    FVector preActivation = fwdProducts.derivative * err;
    FVector backpropagatedError =
        _weight.transposed().multiplyVector(preActivation);
    FLeftMatrix currentWeightDelta =
        preActivation.multiplyTransposed(fwdProducts.input);
    FVector currentBiasDelta = preActivation;
    return BicliqueBackProducts(err, backpropagatedError,
        BicliqueDelta(currentWeightDelta, currentBiasDelta));
  }

  @override
  void updateWeights(Delta delta, double maxWeight, double maxBias) {
    if (delta is BicliqueDelta) {
      _weight.subtract(delta._weight.clamped(-maxWeight, maxWeight));
      //_weight.clamp(-maxWeight, maxWeight);
      _bias.subtract(delta._bias.clamped(-maxBias, maxBias));
      //_bias.clamp(-maxBias, maxBias);
    }
  }

  @override
  Delta zeroDelta() {
    return BicliqueDelta(FLeftMatrix.zero(_weight.nColumns, _weight.nRows),
        FVector.zero(_bias.nRows));
  }
}
