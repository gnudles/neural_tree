import 'package:neural_tree/src/activation_function.dart';
import 'package:neural_tree/src/component.dart';
import 'package:neural_tree/src/linalg.dart';

class WeightBackProducts extends LayerBackwardProducts {
  WeightDelta get delta => abstractDelta as WeightDelta;
  const WeightBackProducts(
      FVector forwardError, FVector propagatedError, WeightDelta delta)
      : super(forwardError, propagatedError, delta);
}

class WeightDelta implements Delta {
  int get outWidth => _weight.length;
  int get inWidth => _weight.length;
  FVector get weight => _weight;
  final FVector _weight;
  WeightDelta(this._weight);

  factory WeightDelta.fromJson(Map<String, dynamic> map) {
    return WeightDelta(FVector.fromJson(map['w']));
  }

  @override
  void add(Delta other) {
    if (other is WeightDelta) {
      this._weight.add(other._weight);
    }
  }

  @override
  void scale(double factor) {
    _weight.scale(factor);
  }

  @override
  void clamp(double maxVal) {
    this._weight.clamp(-maxVal, maxVal);
  }

  @override
  Map<String, dynamic> toJson() {
    return {'type': 'weight', 'w': _weight.toJson()};
  }
}

/// (1,1) Biregular graph (Parallel Pipes)
class WeightComponent extends Component {
  FVector _weight;
  WeightComponent(int width, this._weight) : super(width, width);

  factory WeightComponent.random(
      int width,  double Function() randomWeight) {
    FVector weight = FVector.generate(width, (i) => randomWeight());

    return WeightComponent(width, weight);
  }
  factory WeightComponent.fromJson(Map<String, dynamic> spec) {
    FVector weight = FVector.fromJson(spec["weight"]);
    return WeightComponent(weight.nRows, weight);
  }

  Map<String, dynamic> toJson() {
    return {
      "type": "weight",
      "weight": _weight.toJson(),
    };
  }

  @override
  FVector feedForward(FVector input) {
    return (_weight * input);
    
  }

  @override
  ComponentForwardProducts produce(FVector input) {
    FVector intermediateVector = (_weight * input);
    return OperatorFwdProducts(
        input,
        intermediateVector
        );
  }

  @override
  ComponentBackwardProducts backPropagate(
      ForwardProducts fwdProducts, FVector err) {
    if (!(fwdProducts is OperatorFwdProducts)) throw ArgumentError();
    FVector backpropagatedError = _weight * err;
    FVector currentWeightDelta = (err * fwdProducts.input);
    return WeightBackProducts(err, backpropagatedError,
        WeightDelta(currentWeightDelta));
  }

  @override
  void updateWeights(Delta delta, double maxWeight, double maxBias) {
    if (delta is WeightDelta) {
      _weight.subtract(delta._weight.clamped(-maxWeight, maxWeight));
    }
  }

  @override
  Delta zeroDelta() {
    return WeightDelta(FVector.zero(_weight.nRows));
  }
}
