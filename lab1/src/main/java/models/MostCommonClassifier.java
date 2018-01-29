package models;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Gives predictions on the class of a provided set by training with a test set and classifying based on the most common
 * class witnessed in the training set. If class A was the most prevalent when trained, this classifier will predict that
 * class for every prediction given.
 */
public class MostCommonClassifier extends AbstractClassifier {

    private Integer mostCommon = null;

    @Override
    public void fit(RealMatrix X, RealVector y) {
        super.trained = true;
        this.mostCommon = mostCommon(y);
    }

    @Override
    public RealVector predict(RealMatrix X) {

        if (!super.trained) throw new IllegalStateException("You must train the model with the fit method first!");

        double[][] data = X.getData();
        double[] prediction = new double[data.length];

        for (int i = 0; i < prediction.length; i++) {
            prediction[i] = mostCommon;
        }

        return new ArrayRealVector(prediction);
    }
}
