import 'package:neural_tree/src/activation_function.dart';
import 'package:neural_tree/src/component.dart';
import 'package:neural_tree/src/linalg.dart';
import 'package:neural_tree/src/node.dart';

class UniformBackProducts extends LayerBackwardProducts {
  UniformDelta get delta => abstractDelta as UniformDelta;
  const UniformBackProducts(
      FVector forwardError, FVector propagatedError, UniformDelta delta)
      : super(forwardError, propagatedError, delta);
}

class UniformDelta implements Delta {
  int get outWidth => 1;
  int get inWidth => 1;
  double get weight => _weight;
  double get bias => _bias;
  double _weight;
  double _bias;
  UniformDelta(this._weight, this._bias);
  factory UniformDelta.fromJson(Map<String, dynamic> map) {
    return UniformDelta(map['w'], map['b']);
  }

  @override
  void add(Delta other) {
    if (other is UniformDelta) {
      _weight += other._weight;
      _bias += other._bias;
    }
  }

  @override
  void scale(double factor) {
    _weight *= factor;
    _bias *= factor;
  }

  @override
  Map<String, dynamic> toJson() {
    return {'type': 'uniform', 'w': _weight, 'b': _bias};
  }
}

/// every lane get the same transformation
class UniformComponent extends Component {
  ActivationFunction _activationFunc;
  double _weight;
  double _bias;
  UniformComponent(int width, this._weight, this._bias,
      ActivationFunctionType activationFuncType)
      : _activationFunc = mapActivationFunction[activationFuncType]!,
        super(width, width);

  factory UniformComponent.random(
      int width,
      ActivationFunctionType activationFuncType,
      double Function() randomBias,
      double Function() randomWeight) {
    double weight = randomWeight();
    double bias = randomBias();
    return UniformComponent(width, weight, bias, activationFuncType);
  }
  factory UniformComponent.fromJson(Map<String, dynamic> map) {
    double weight = map["weight"];
    double bias = map["bias"];
    int width = map["width"];
    ActivationFunctionType activationFuncType =
        activationTypeFromString[map["activation"]]!;
    return UniformComponent(width, weight, bias, activationFuncType);
  }

  Map<String, dynamic> toJson() {
    return {
      "type": "uniform",
      "width": inWidth,
      "weight": _weight,
      "bias": _bias,
      "activation": _activationFunc.type.toString()
    };
  }

  @override
  FVector feedForward(FVector input) {
    return input.scaled(_weight)
      ..addScalar(_bias)
      ..apply(_activationFunc.func, _activationFunc.funcSIMD);
  }

  @override
  ComponentForwardProducts produce(FVector input) {
    FVector intermediateVector = input.scaled(_weight)..addScalar(_bias);
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
    FVector backpropagatedError = preActivation.scaled(_weight);
    double lengthReciprocal = 1 / preActivation.length;
    double currentWeightDelta =
        (preActivation * fwdProducts.input).sumElements() * lengthReciprocal;
    double currentBiasDelta = preActivation.sumElements() * lengthReciprocal;
    return UniformBackProducts(err, backpropagatedError,
        UniformDelta(currentWeightDelta, currentBiasDelta));
  }

  @override
  void updateWeights(Delta delta, double maxWeight, double maxBias) {
    if (delta is UniformDelta) {
      _weight -= delta._weight;
      _weight.clamp(-maxWeight, maxWeight);
      _bias -= delta._bias;
      _bias.clamp(-maxBias, maxBias);
    }
  }

  @override
  Delta zeroDelta() {
    return UniformDelta(0, 0);
  }
}
