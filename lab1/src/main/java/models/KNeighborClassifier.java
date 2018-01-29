package models;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.*;
import java.util.function.Function;

/**
 * Uses the KNeighbors algorithm to predict a matrix of X instances by finding training instance(s) that most closely
 * match X.
 */
public class KNeighborClassifier extends AbstractClassifier {

    private Function<RealMatrix, Double> weightFunction;
    private int k;
    private RealMatrix XTrain;
    private RealVector yTrain;

    /**
     * K is the number of training instances that will have an input in determining X's predicted class.
     *
     * @param k the number of neighbors polled
     * @param weightFunction a function determining the significance of a neighbor
     */
    public KNeighborClassifier(int k, Function<RealMatrix, Double> weightFunction) {
        super();

        this.k = k;
        this.weightFunction = weightFunction;
    }

    @Override
    public void fit(RealMatrix X, RealVector y) {
        this.XTrain = X.copy();
        this.yTrain = y.copy();
        super.trained = true;
    }

    @Override
    public RealVector predict(RealMatrix X) {

        if (!super.trained) throw new IllegalStateException("You must train the model with the fit method first!");

        double[] rowTest, rowTrain, predictions;
        double temp1;
        Integer prediction;
        List<Double> predictionsList = new LinkedList<>();

        Queue<Map.Entry<Integer, Double>> ranking = new PriorityQueue<>((o1, o2) -> {
            double difference = (o2.getValue() - o1.getValue());
            if (difference > 0) return 1;
            else if (difference < 0) return -1;
            return 0;
        });

        for (int i = 0; i < X.getRowDimension(); i++) {
            rowTest = X.getRow(i);
            for (int j = 0; j < XTrain.getRowDimension(); j++) {
                rowTrain = XTrain.getRow(j);
                temp1 = weightFunction.apply(new Array2DRowRealMatrix(new double[][] {rowTest, rowTrain}));
                ranking.add(new AbstractMap.SimpleEntry<>(j, temp1));
            }

            predictions = new double[k];
            for (int j = 0; j < k; j++) {
                predictions[j] = yTrain.getEntry(ranking.poll().getKey());
            }
            prediction = mostCommon(new ArrayRealVector(predictions));
            ranking.clear();

            predictionsList.add(prediction.doubleValue());
        }

        Double[] temp2 = new Double[predictionsList.size()];
        predictionsList.toArray(temp2);

        return new ArrayRealVector(temp2);
    }
}
