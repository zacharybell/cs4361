package models;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class MostCommonClassifier extends AbstractClassifier {

    private Integer mostCommon = null;

    @Override
    public void fit(RealMatrix X, RealVector y) {
        Map<Integer, Integer> yCount = new HashMap<>();
        int temp, count;

        double[] yTest = y.toArray();
        for (int i = 0; i < yTest.length; i++) {
             temp = (int) Math.round(yTest[i]);

             if (yCount.containsKey(temp)) {
                 count = yCount.get(temp);
                 yCount.put(temp, ++count);
             }
             else {
                 yCount.put(temp, 1);
             }
        }

        Iterator<Integer> keys = yCount.keySet().iterator();

        int key, value, max = Integer.MIN_VALUE;
        Integer maxKey = null;

        while (keys.hasNext()) {
            key = keys.next();
            value = yCount.get(key);
            if (value > max) {
                max = value;
                maxKey = key;
            }
        }

        super.trained = true;
        this.mostCommon = maxKey;
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
