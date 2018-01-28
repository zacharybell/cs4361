package models;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public abstract class AbstractClassifier {

    protected boolean trained = false;

    protected abstract void fit(RealMatrix X, RealVector y);
    protected abstract RealVector predict(RealMatrix X);
}
