package com.github.zubmike.nn;

import java.io.*;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class App {

	private static final String NEURAL_NETWORK_FILE_NAME = "neural-network.ser";
	
	private static final double AND = 1;
	private static final double OR =  2;
	private static final double XOR = 3;
	private static final Map<Double, String> TYPE_MAP = new LinkedHashMap<>();
	static {
		TYPE_MAP.put(AND, "AND");
		TYPE_MAP.put(OR, " OR");
		TYPE_MAP.put(XOR, "XOR");
	}

	private static final Map<List<Double>, Double> LEARNING_MAP = new LinkedHashMap<>();
	static {
		LEARNING_MAP.put(List.of(AND, 0.0, 0.0), 0.0);
		LEARNING_MAP.put(List.of(AND, 0.0, 1.0), 0.0);
		LEARNING_MAP.put(List.of(AND, 1.0, 0.0), 0.0);
		LEARNING_MAP.put(List.of(AND, 1.0, 1.0), 1.0);

		LEARNING_MAP.put(List.of(OR, 0.0, 0.0), 0.0);
		LEARNING_MAP.put(List.of(OR, 0.0, 1.0), 1.0);
		LEARNING_MAP.put(List.of(OR, 1.0, 0.0), 1.0);
		LEARNING_MAP.put(List.of(OR, 1.0, 1.0), 1.0);

		LEARNING_MAP.put(List.of(XOR, 0.0, 0.0), 0.0);
		LEARNING_MAP.put(List.of(XOR, 0.0, 1.0), 1.0);
		LEARNING_MAP.put(List.of(XOR, 1.0, 0.0), 1.0);
		LEARNING_MAP.put(List.of(XOR, 1.0, 1.0), 0.0);
	}

	public static void main(String[] args) {
		process(true);
	}

	private static void process(boolean recreate) {
		var neuralNetwork = recreate ? createNeuralNetwork() : loadNeuralNetwork();
		print(neuralNetwork);
	}

	private static NeuralNetwork createNeuralNetwork() {
		var neuralNetwork = new NeuralNetwork();
		neuralNetwork.learn(LEARNING_MAP);
		serialize(neuralNetwork);
		return neuralNetwork;
	}

	private static NeuralNetwork loadNeuralNetwork() {
		return deserialize();
	}

	private static void serialize(NeuralNetwork neuralNetwork) {
		try (var fileOutputStream = new FileOutputStream(App.NEURAL_NETWORK_FILE_NAME);
			 var objectOutputStream = new ObjectOutputStream(fileOutputStream)) {
			objectOutputStream.writeObject(neuralNetwork);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	private static NeuralNetwork deserialize() {
		try (var fileInputStream = new FileInputStream(App.NEURAL_NETWORK_FILE_NAME);
			 var objectInputStream = new ObjectInputStream(fileInputStream)) {
			return (NeuralNetwork) objectInputStream.readObject();
		} catch (ClassNotFoundException | IOException e) {
			throw new RuntimeException(e);
		}
	}

	private static void print(NeuralNetwork neuralNetwork) {
		LEARNING_MAP.keySet().forEach(inputs -> {
			System.out.println(Math.round(inputs.get(1)) + " " +
					TYPE_MAP.get(inputs.get(0)) + " " +
					Math.round(inputs.get(2)) + " = " +
					Math.round(neuralNetwork.output(inputs)));
		});
	}

}
