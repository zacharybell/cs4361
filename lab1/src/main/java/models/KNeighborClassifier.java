package models;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;

public class KNeighborClassifier extends AbstractClassifier {

    private Function<RealMatrix, Double> weightFunction;
    private int k;
    private RealMatrix XTrain;
    private RealVector yTrain;

    KNeighborClassifier(int k, Function<RealMatrix, Double> weightFunction) {
        super();

        this.k = k;
        this.weightFunction = weightFunction;
    }


    @Override
    protected void fit(RealMatrix X, RealVector y) {
        this.XTrain = X.copy();
        this.yTrain = y.copy();
        super.trained = true;
    }

    @Override
    protected RealVector predict(RealMatrix X) {

        double[] rowTest, rowTrain;
        double min = Double.MAX_VALUE, temp;
        Double prediction = null;
        List<Double> predictionsList = new LinkedList<>();

        for (int i = 0; i < X.getRowDimension(); i++) {
            rowTest = X.getRow(i);
            for (int j = 0; j < XTrain.getRowDimension(); j++) {
                rowTrain = XTrain.getRow(j);
                temp = weightFunction.apply(new Array2DRowRealMatrix(new double[][] {rowTest, rowTrain}));
                if (temp < min) {
                    prediction = yTrain.getEntry(j);
                    min = temp;
                }
            }
            predictionsList.add(prediction);
        }

        Double[] predictions = new Double[predictionsList.size()];
        predictionsList.toArray(predictions);

        return new ArrayRealVector(predictions);
    }
}
