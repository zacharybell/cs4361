package models;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public abstract class AbstractClassifier {

    protected boolean trained = false;

    protected abstract void fit(@NotNull RealMatrix X, @NotNull RealVector y);
    protected abstract RealVector predict(@NotNull RealMatrix X);

    /**
     * Gets the most common value in a vector. The RealVector contains doubles and the double values are rounded to the
     * closest integer. The most commonly occurring integer is thus returned.
     *
     * @param vector a RealVector of values
     * @return the most commonly occurring value(rounded)
     */
    protected Integer mostCommon(@NotNull RealVector vector) {
        Map<Integer, Integer> counter = new HashMap<>();
        int temp, count;

        double[] yTest = vector.toArray();
        for (int i = 0; i < yTest.length; i++) {
            temp = (int) Math.round(yTest[i]);

            if (counter.containsKey(temp)) {
                count = counter.get(temp);
                counter.put(temp, ++count);
            }
            else {
                counter.put(temp, 1);
            }
        }

        Iterator<Integer> keys = counter.keySet().iterator();

        int key, value, max = Integer.MIN_VALUE;
        Integer maxKey = null;

        while (keys.hasNext()) {
            key = keys.next();
            value = counter.get(key);
            if (value > max) {
                max = value;
                maxKey = key;
            }
        }

        return maxKey;
    }
}
