import 'dart:typed_data';

import 'package:neural_tree/src/activation_function.dart';
import 'package:neural_tree/src/biclique.dart';
import 'package:neural_tree/src/biregular.dart';
import 'package:neural_tree/src/graph_component.dart';
import 'package:neural_tree/src/uniform.dart';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';

class ActivationNode extends SingleInputNode {
  ActivationFunctionType activation;
  ActivationNode(this.activation, Node input, {String name = ""})
      : super(input, input.outWidth, name);

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    return ActivationNodeImpl(id, name, outWidth, dependencies, activation);
  }
}

class ActivationNodeImpl extends NodeImpl {
  ActivationFunctionType activation;
  ActivationFunction activationFunction;

  ActivationNodeImpl(int id, String name, int outWidth, Int32List dependencies,
      this.activation)
      : activationFunction = mapActivationFunction[activation]!,
        super(id, name, outWidth, dependencies);

  factory ActivationNodeImpl.fromJson(Map<String, dynamic> map) {
    int id = map['id'];
    //int outWidth = map['outWidth'];
    Int32List dependencies =
        Int32List.fromList((map['dependencies'] as List<dynamic>).cast());
    String name = map['name'];
    int outWidth = map['outWidth'];
    ActivationFunctionType activation =
        activationTypeFromString[map['activation']]!;

    return ActivationNodeImpl(id, name, outWidth, dependencies, activation);
  }
  @override
  Map<String, dynamic> toJson() {
    return {
      'node_type': 'activation',
      'id': id,
      'name': name,
      'outWidth': outWidth,
      'dependencies': dependencies,
      'activation': activation.toString(),
    };
  }

  @override
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts) {
    FVector input = feeds[dependencies[0]]!;
    FVector result;
    result =
        input.applied(activationFunction.func, activationFunction.funcSIMD);
    feeds[id] = result;
    if (fwdProducts != null) {
      fwdProducts[id] = LayerFwdProducts(
          input,
          result,
          input.applied(activationFunction.derivative,
              activationFunction.derivativeSIMD));
    }
  }

  @override
  void backPropagate(List<BackwardProducts> backProducts, List<Delta?> deltas,
      List<FVector?> propagatedErrors, List<ForwardProducts> fwdProducts) {
    var resultErr = (fwdProducts[id] as LayerFwdProducts).derivative * propagatedErrors[id]!;
    
    
    var depId = dependencies.single;
    if (propagatedErrors[depId] == null) {
      propagatedErrors[depId] = resultErr;
    } else {
      propagatedErrors[depId]!.add(resultErr);
    }
  }
    
  @override
  void update(Delta? delta, double maxWeight, double maxBias) {
  }
}
