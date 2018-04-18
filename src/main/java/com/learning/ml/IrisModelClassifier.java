package com.learning.ml;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;


public class IrisModelClassifier implements ModelClassifier{

    private Attribute petallength;
    private Attribute petalwidth;

    private ArrayList<Attribute> attributes;
    private ArrayList<String> classVal;
    private Instances dataRaw;


    public IrisModelClassifier() {
        petallength = new Attribute("petallength");
        petalwidth = new Attribute("petalwidth");
        attributes = new ArrayList<Attribute>();
        classVal = new ArrayList<String>();
        classVal.add("Iris-setosa");
        classVal.add("Iris-versicolor");
        classVal.add("Iris-virginica");

        attributes.add(petallength);
        attributes.add(petalwidth);

        attributes.add(new Attribute("class", classVal));
        dataRaw = new Instances("TestInstances", attributes, 0);
        dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
    }

    @Override
    public Instances createInstance(double petallength, double petalwidth, double result) {
        dataRaw.clear();
        double[] instanceValue1 = new double[]{petallength, petalwidth, 0};
        dataRaw.add(new DenseInstance(1.0, instanceValue1));
        return dataRaw;
    }


    public String classify(Instances insts, String path) {
        String result = "Not classified!!";
        Classifier cls = null;
        try {
            cls = (MultilayerPerceptron) SerializationHelper.read(path);
            result = classVal.get((int) cls.classifyInstance(insts.firstInstance()));
        } catch (Exception ex) {
            Logger.getLogger(IrisModelClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        return result;
    }


    public Instances getInstance() {
        return dataRaw;
    }

	


}
