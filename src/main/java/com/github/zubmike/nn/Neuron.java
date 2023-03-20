package com.github.zubmike.nn;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public abstract class Neuron implements Serializable {

	@Serial
	private static final long serialVersionUID = 7537001228180549725L;

	private final List<Double> inputs;
	private final List<Double> weights;
	private double bias;
	private Double output;

	public Neuron(int inputSize) {
		inputs = new ArrayList<>(inputSize);
		weights = new ArrayList<>(inputSize);
		IntStream.range(0, inputSize)
				.mapToDouble(i -> Math.random())
				.forEach(weights::add);
		bias = Math.random();
	}

	public void input(double... values) {
		inputs.clear();
		DoubleStream.of(values).forEach(inputs::add);
		output = null;
	}

	public void input(Collection<Double> values) {
		inputs.clear();
		inputs.addAll(values);
		output = null;
	}

	public double learn(double rate, double outputError) {
		var error = calcDerivative(output()) * outputError;
		for (int i = 0; i < weights.size(); i++) {
			var newWeight = weights.get(i) + rate * error * inputs.get(i);
			weights.set(i, newWeight);
		}
		bias += rate * error;
		output = null;
		return error;
	}

	public double output() {
		if (output == null) {
			var sum = 0.0;
			for (int i = 0; i < inputs.size(); i++) {
				sum += inputs.get(i) * weights.get(i);
			}
			output = calc(sum + bias);
		}
		return output;
	}

	protected abstract double calc(double value);

	protected abstract double calcDerivative(double value);

	public double getWeight(int inputIndex) {
		return weights.get(inputIndex);
	}

}
