package com.github.zubmike.nn;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;
import java.util.stream.DoubleStream;

public class NeuralNetwork implements Serializable {

	@Serial
	private static final long serialVersionUID = 8193960225880145525L;

	private static final int INPUT_SIZE = 3;
	private static final int HIDDEN_LAYER_SIZE = 3;
	private static final int OUTPUT_LAYER_NEURON_INDEX = HIDDEN_LAYER_SIZE;

	private static final double LEARNING_RATE = 0.1;
	private static final int EPOCHS = 100000;

	private final List<Neuron> neurons = new ArrayList<>();

	public NeuralNetwork() {
		initNeurons();
	}

	private void initNeurons() {
		for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
			neurons.add(new SigmoidNeuron(INPUT_SIZE));
		}
		neurons.add(new SigmoidNeuron(INPUT_SIZE));
	}

	public void learn(Map<List<Double>, Double> learningMap) {
		for (int i = 0; i < EPOCHS; i++) {
			learningMap.forEach(this::learn);
		}
	}

	private void learn(Collection<Double> inputs, Double target) {
		var output = output(inputs);
		var error = target - output;
		var outputNeuron = neurons.get(OUTPUT_LAYER_NEURON_INDEX);
		var outputError = outputNeuron.learn(LEARNING_RATE, error);
		for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
			neurons.get(i).learn(LEARNING_RATE,
					outputNeuron.getWeight(i) * outputError);
		}
	}

	public double output(double... inputs) {
		return output(DoubleStream.of(inputs).boxed().toList());
	}

	public double output(Collection<Double> inputs) {
		if (inputs.size() > INPUT_SIZE) {
			throw new IllegalArgumentException("Invalid input size");
		}
		var hiddenLayerOutputs = new ArrayList<Double>();
		for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
			var neuron = neurons.get(i);
			neuron.input(inputs);
			hiddenLayerOutputs.add(neuron.output());
		}
		neurons.get(OUTPUT_LAYER_NEURON_INDEX)
				.input(hiddenLayerOutputs);
		return neurons.get(OUTPUT_LAYER_NEURON_INDEX).output();
	}

}
