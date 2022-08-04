import 'package:neural_tree/src/activation_function.dart';
import 'package:neural_tree/src/component.dart';
import 'package:neural_tree/src/linalg.dart';

class BiasBackProducts extends LayerBackwardProducts {
  BiasDelta get delta => abstractDelta as BiasDelta;
  const BiasBackProducts(
      FVector forwardError, FVector propagatedError, BiasDelta delta)
      : super(forwardError, propagatedError, delta);
}

class BiasDelta implements Delta {
  int get outWidth => _bias.length;
  int get inWidth => _bias.length;
  FVector get bias => _bias;
  final FVector _bias;
  BiasDelta(this._bias);

  factory BiasDelta.fromJson(Map<String, dynamic> map) {
    return BiasDelta(FVector.fromJson(map['b']));
  }

  @override
  void add(Delta other) {
    if (other is BiasDelta) {
      this._bias.add(other._bias);
    }
  }

  @override
  void scale(double factor) {
    _bias.scale(factor);
  }

  @override
  void clamp(double maxVal) {
    _bias.clamp(-maxVal, maxVal);
  }

  @override
  double minAbsDelta() {
    return _bias.abs().smallestElement();
  }

  @override
  Map<String, dynamic> toJson() {
    return {'type': 'bias', 'b': _bias.toJson()};
  }
}

class BiasComponent extends Component {
  FVector _bias;
  BiasComponent(int width, this._bias) : super(width, width);

  factory BiasComponent.random(int width, double Function() randomBias) {
    FVector bias = FVector.generate(width, (i) => randomBias());
    return BiasComponent(width, bias);
  }
  factory BiasComponent.fromJson(Map<String, dynamic> spec) {
    FVector bias = FVector.fromJson(spec["bias"]);
    return BiasComponent(bias.nRows, bias);
  }

  Map<String, dynamic> toJson() {
    return {
      "type": "bias",
      "bias": _bias.toJson(),
    };
  }

  @override
  FVector feedForward(FVector input) {
    return (input + _bias);
  }

  @override
  ComponentForwardProducts produce(FVector input) {
    FVector intermediateVector = input + _bias;
    return OperatorFwdProducts(input, intermediateVector);
  }

  @override
  ComponentBackwardProducts backPropagate(
      ForwardProducts fwdProducts, FVector err) {
    if (!(fwdProducts is OperatorFwdProducts)) throw ArgumentError();
    return BiasBackProducts(err, err, BiasDelta(err));
  }

  @override
  void updateWeights(Delta delta, double maxWeight, double maxBias) {
    if (delta is BiasDelta) {
      _bias.subtract(delta._bias.clamped(-maxBias, maxBias));
      //_bias.clamp(-maxBias, maxBias);
    }
  }

  @override
  Delta zeroDelta() {
    return BiasDelta(FVector.zero(_bias.nRows));
  }
}
