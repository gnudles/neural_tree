import 'package:neural_tree/src/activation_function.dart';
import 'package:neural_tree/src/linalg.dart';
import 'package:neural_tree/src/multi_graph.dart';
import 'package:neural_tree/src/node.dart';

import 'biclique.dart';
import 'biregular.dart';
import 'graph_component.dart';
import 'uniform.dart';

/// The delta of weights and bias, when updating a matrix
abstract class Delta {
  void add(Delta other);
  void scale(double factor);
  Map<String, dynamic> toJson();
  static Delta? fromJson(Map<String, dynamic> map) {
    var initMapping = <String, Delta Function(Map<String, dynamic>)>{
      'graph': (map) => DeltaList.fromJson(map),
      'biregular': (map) => BiregularDelta.fromJson(map),
      'biclique': (map) => BicliqueDelta.fromJson(map),
      'uniform': (map) => UniformDelta.fromJson(map),
    };
    if (!map.containsKey('type') || !initMapping.containsKey(map['type'])) {
      return null;
    }

    return initMapping[map['type']]!(map);
  }
}

/// Forward Products
///
/// These are usually the activation for the next layer, and the derivative.
class ForwardProducts {
  const ForwardProducts();
}

class ComponentForwardProducts extends ForwardProducts {
  const ComponentForwardProducts();
  FVector get output => throw UnimplementedError();
}

class BackwardProducts {
  const BackwardProducts();
}

class ComponentBackwardProducts extends BackwardProducts {
  const ComponentBackwardProducts();
  FVector get propagatedError {
    throw UnimplementedError();
  }

  Delta get abstractDelta {
    throw UnimplementedError();
  }
}

/// backward Products
///
/// These are usually the input error, output error,
class LayerBackwardProducts extends ComponentBackwardProducts {
  @override
  FVector get propagatedError => _propagatedError;
  final FVector _propagatedError;
  final FVector _forwardError;
  final Delta _delta;
  Delta get abstractDelta => _delta;
  const LayerBackwardProducts(
      this._forwardError, this._propagatedError, this._delta);
}

class LayerFwdProducts extends ComponentForwardProducts {
  @override
  FVector get output => activation;
  final FVector input;
  final FVector activation;
  final FVector derivative;
  const LayerFwdProducts(this.input, this.activation, this.derivative);
}

abstract class Component {
  final int _inWidth;
  final int _outWidth;
  int get inWidth => _inWidth;
  int get outWidth => _outWidth;
  Component(this._inWidth, this._outWidth);
  FVector feedForward(FVector input);
  ComponentForwardProducts produce(FVector input);
  ComponentBackwardProducts backPropagate(
      ForwardProducts fwdProducts, FVector err);
  void updateWeights(Delta delta);
  Delta zeroDelta();
  Map<String, dynamic> toJson();
}

Map<String, Component Function(Map<String, dynamic>)> componentLoaders = {
  'graph': (map) => GraphComponent.fromJson(map),
  'uniform': (map) => UniformComponent.fromJson(map),
  'biregular': (map) => BiregularComponent.fromJson(map),
  'biclique': (map) => BicliqueComponent.fromJson(map),
};
