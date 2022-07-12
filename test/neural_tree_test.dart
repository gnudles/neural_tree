import 'dart:math';

import 'package:neural_tree/neural_tree.dart';
import 'package:test/test.dart';

void main() {
  group('A group of tests', () {
    test('First Test', () {
      InputNode in1 = InputNode(2, "xor_in");
      Random r = Random();
      var rgen = () => r.nextDouble() - 0.5;
      var c1 = ComponentNode(
          BicliqueComponent.random(
              in1.outWidth, 2, ActivationFunctionType.uacsls, rgen, rgen),
          in1);
      var c2 = ComponentNode(
          BiregularComponent.random(
              c1.outWidth, ActivationFunctionType.uacsls, rgen, rgen),
          c1);
      var c3 = ComponentNode(
          BicliqueComponent.random(
              c2.outWidth, 1, ActivationFunctionType.cubicSigmoid, rgen, rgen),
          c2);
      MultiGraph g = MultiGraph.fromOutputNodes([c3], [in1]);
      expect(g.executeChain.length == 3, isTrue);
      expect(g.totalInputs == 1, isTrue);
      expect(g.totalOutputs == 1, isTrue);
      expect(g.totalNodes == 4, isTrue);
      print(g.feedForward([FVector.filled(2, -1.0)]).single.listView);
      print(g
          .feedForward([
            FVector.fromList([1.0, 1.0])
          ])
          .single
          .listView);
      print(g
          .feedForward([
            FVector.fromList([-1.0, 1.0])
          ])
          .single
          .listView);
      var inputs = [
        FVector.fromList([1.0, 1.0]),
        FVector.fromList([1.0, -1.0]),
        FVector.fromList([-1.0, 1.0]),
        FVector.fromList([-1.0, -1.0])
      ];
      var outputs = [
        FVector.fromList([-1.0]),
        FVector.fromList([1.0]),
        FVector.fromList([1.0]),
        FVector.fromList([-1.0])
      ];
      List<double> learningRates = [0.4, 0.15, 0.01, 0.004];
      const momentum = 0.96;
      for (var lRate in learningRates) {
        double maxError=1000;
        DeltaList delta = g.zeroDelta(); // zero delta.
        for (int i = 0; i < 10000 || maxError*2>lRate; ++i) {
          DeltaList newDelta = g.zeroDelta(); // zero delta.
          maxError = 0;
          for (int s = 0; s < 4; s++) {
            var fwdFlow = g.produce([inputs[s]]);
            var bckFlow = g.backPropagateByTarget(fwdFlow, [outputs[s]]);
            var error =
                bckFlow.propagatedErrors[g.outputIndices.single]!.single;
            error *= error;
            if (error > maxError) maxError = error;
            bckFlow.delta.scale(lRate * (1 + 0.2 * rgen()));
            newDelta.add(bckFlow.delta);
          }
          newDelta.scale(lRate * 0.25);
          delta.scale(momentum);
          delta.add(newDelta);
          g.update(delta);
        }
      }
      print(g.feedForward([FVector.filled(2, -1.0)]).single.listView);
      print(g
          .feedForward([
            FVector.fromList([1.0, 1.0])
          ])
          .single
          .listView);
      print(g
          .feedForward([
            FVector.fromList([-1.0, 1.0])
          ])
          .single
          .listView);
      print(g
          .feedForward([
            FVector.fromList([1.0, -1.0])
          ])
          .single
          .listView);
    });
  });
}
