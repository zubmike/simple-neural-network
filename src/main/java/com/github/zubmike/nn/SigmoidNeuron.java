package com.github.zubmike.nn;

import java.io.Serial;

public class SigmoidNeuron extends Neuron {

	@Serial
	private static final long serialVersionUID = -7579432123086607611L;

	public SigmoidNeuron(int inputSize) {
		super(inputSize);
	}

	@Override
	protected double calc(double value) {
		return 1 / (1 + Math.exp(-value));
	}

	@Override
	protected double calcDerivative(double value) {
		return value * (1 - value);
	}

}
