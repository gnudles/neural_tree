import 'dart:typed_data';

import 'package:neural_tree/neural_tree.dart';
import 'package:neural_tree/src/activation_function.dart';
import 'package:neural_tree/src/biclique.dart';
import 'package:neural_tree/src/biregular.dart';
import 'package:neural_tree/src/graph_component.dart';
import 'package:neural_tree/src/nodes/sum_node.dart';
import 'package:neural_tree/src/uniform.dart';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';
Node createBiregularProjector(Node input,Iterable<ActivationFunctionType> activations, double Function() randomBias,
      double Function() randomWeight)
{
  
  return SumNode(activations.map((act)
  {
    var c1 = ComponentNode(BiregularComponent.random(input.outWidth, act, randomBias, randomWeight),input);
    return ComponentNode(WeightComponent.random(c1.outWidth, randomWeight),c1);
  } ).toList());
}