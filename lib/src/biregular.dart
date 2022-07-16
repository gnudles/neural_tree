import 'package:neural_tree/src/activation_function.dart';
import 'package:neural_tree/src/component.dart';
import 'package:neural_tree/src/linalg.dart';

class BiregularBackProducts extends LayerBackwardProducts {
  BiregularDelta get delta => abstractDelta as BiregularDelta;
  const BiregularBackProducts(
      FVector forwardError, FVector propagatedError, BiregularDelta delta)
      : super(forwardError, propagatedError, delta);
}

class BiregularDelta implements Delta {
  int get outWidth => _bias.length;
  int get inWidth => _bias.length;
  FVector get weight => _weight;
  FVector get bias => _bias;
  final FVector _weight;
  final FVector _bias;
  BiregularDelta(this._weight, this._bias);

  factory BiregularDelta.fromJson(Map<String, dynamic> map) {
    return BiregularDelta(
        FVector.fromJson(map['w']), FVector.fromJson(map['b']));
  }

  @override
  void add(Delta other) {
    if (other is BiregularDelta) {
      this._bias.add(other._bias);
      this._weight.add(other._weight);
    }
  }

  @override
  void scale(double factor) {
    _weight.scale(factor);
    _bias.scale(factor);
  }

  @override
  Map<String, dynamic> toJson() {
    return {'type': 'biregular', 'w': _weight.toJson(), 'b': _bias.toJson()};
  }
}

/// (1,1) Biregular graph (Parallel Pipes)
class BiregularComponent extends Component {
  ActivationFunction _activationFunc;
  FVector _weight;
  FVector _bias;
  BiregularComponent(int width, this._weight, this._bias,
      ActivationFunctionType activationFuncType)
      : _activationFunc = mapActivationFunction[activationFuncType]!,
        super(width, width);

  factory BiregularComponent.random(
      int width,
      ActivationFunctionType activationFuncType,
      double Function() randomBias,
      double Function() randomWeight) {
    FVector weight = FVector.generate(width, (i) => randomWeight());
    FVector bias = FVector.generate(width, (i) => randomBias());
    return BiregularComponent(width, weight, bias, activationFuncType);
  }
  factory BiregularComponent.fromJson(Map<String, dynamic> spec) {
    FVector weight = FVector.fromJson(spec["weight"]);
    FVector bias = FVector.fromJson(spec["bias"]);
    ActivationFunctionType activationFuncType =
        activationTypeFromString[spec["activation"]]!;
    return BiregularComponent(weight.nRows, weight, bias, activationFuncType);
  }

  Map<String, dynamic> toJson() {
    return {
      "type": "biregular",
      "weight": _weight.toJson(),
      "bias": _bias.toJson(),
      "activation": _activationFunc.type.toString()
    };
  }

  @override
  FVector feedForward(FVector input) {
    return ((_weight * input) + _bias)
      ..apply(_activationFunc.func, _activationFunc.funcSIMD);
  }

  @override
  ComponentForwardProducts produce(FVector input) {
    FVector intermediateVector = (_weight * input) + _bias;
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
    FVector backpropagatedError = _weight * preActivation;
    FVector currentWeightDelta = (preActivation * fwdProducts.input);
    FVector currentBiasDelta = preActivation;
    return BiregularBackProducts(err, backpropagatedError,
        BiregularDelta(currentWeightDelta, currentBiasDelta));
  }

  @override
  void updateWeights(Delta delta, double maxWeight, double maxBias) {
    if (delta is BiregularDelta) {
      _weight.subtract(delta._weight);
      _weight.clamp(-maxWeight, maxWeight);
      _bias.subtract(delta._bias);
      _bias.clamp(-maxBias, maxBias);
    }
  }

  @override
  Delta zeroDelta() {
    return BiregularDelta(
        FVector.zero(_weight.nRows), FVector.zero(_bias.nRows));
  }
}
