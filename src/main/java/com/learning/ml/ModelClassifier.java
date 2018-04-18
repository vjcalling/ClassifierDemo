package com.learning.ml;

import weka.core.Instances;

public interface ModelClassifier {

	Instances createInstance(double petallength, double petalwidth, double result);

	String classify(Instances useFilter, String modelpath);

}
