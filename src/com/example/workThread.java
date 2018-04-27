package com.example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class workThread extends Thread {

    public int mz[];
    private int batch;
    public double alpha, beta;
    private int index;
    public double nzw[][];
    public double nz[];
    public ArrayList<ArrayList<Integer>> docList;
    public int vocSize;
    public int numDoc;
    public int[] assignment;
    private Random rg;
    private int numTopic;
    public ArrayList<ArrayList<ArrayList<Integer>>> wordGPUInfo;
    public ArrayList<ArrayList<Boolean>> wordGPUFlag;
    public double[][] topicProbabilityGivenWord;
    private Map<Integer,ArrayList<Integer>> schemaMap;
    private double weight;

    public workThread(int index,int batch,int numTopic,int[] mz,double[] nz,double[][] nzw,
                      ArrayList<ArrayList<Integer>> docList,double alpha,double beta,
                      double weight,int numDoc,int vocSize,ArrayList<ArrayList<ArrayList<Integer>>> wordGPUInfo,
                      ArrayList<ArrayList<Boolean>> wordGPUFlag,
                      Map<Integer,ArrayList<Integer>> schemaMap,double[][] topicProbabilityGivenWord,int[] ass) {
        this.batch = batch;
        this.assignment = ass.clone();
        this.index = index;
        this.mz = mz.clone();
        this.nzw = nzw.clone();
        this.nz = nz.clone();
        this.docList = docList;
        this.numTopic = numTopic;
        rg = new Random();
        this.alpha = alpha;
        this.beta = beta;
        this.numDoc = numDoc;
        this.vocSize = vocSize;
        this.schemaMap = schemaMap;
        this.weight = weight;
        this.topicProbabilityGivenWord = topicProbabilityGivenWord.clone();
        this.wordGPUFlag = wordGPUFlag;
        this.wordGPUInfo = wordGPUInfo;
    }


    public double findTopicMaxProbabilityGivenWord(int wordID) {
        double max = -1.0;
        for (int i = 0; i < numTopic; i++) {
            double tmp = topicProbabilityGivenWord[wordID][i];
            if (Double.compare(tmp, max) > 0) {
                max = tmp;
            }
        }
        return max;
    }

    public double getTopicProbabilityGivenWord(int topic, int wordID) {
        return topicProbabilityGivenWord[wordID][topic];
    }

    /**
     * update GPU flag, which decide whether do GPU operation or not
     * @param docID
     * @param newTopic
     */
    public void updateWordGPUFlag(int docID, int newTopic) {
        // we calculate the p(t|w) and p_max(t|w) and use the ratio to decide we
        // use gpu for the word under this topic or not
        ArrayList<Integer> termIDArray = docList.get(docID);
        ArrayList<Boolean> docWordGPUFlag = new ArrayList<Boolean>();
        for (int t = 0; t < termIDArray.size(); t++) {

            int termID = termIDArray.get(t);
            double maxProbability = findTopicMaxProbabilityGivenWord(termID);
            double ratio = getTopicProbabilityGivenWord(newTopic, termID) / maxProbability;

            double a = rg.nextDouble();
            docWordGPUFlag.add(Double.compare(ratio, a) > 0);
        }
        wordGPUFlag.set(docID, docWordGPUFlag);
    }

    public void ratioCount(Integer topic, Integer docID, ArrayList<Integer> termIDArray, int flag) {
        mz[topic] += flag;
        for (int t = 0; t < termIDArray.size(); t++) {
            int wordID = termIDArray.get(t);
            nzw[topic][wordID] += flag;
            nz[topic] += flag;
        }
        // we update gpu flag for every document before it change the counter
        // when adding numbers
        if (flag > 0) {
            updateWordGPUFlag(docID, topic);
            for (int t = 0; t < termIDArray.size(); t++) {
                int wordID = termIDArray.get(t);
                boolean gpuFlag = wordGPUFlag.get(docID).get(t);
                ArrayList<Integer> gpuInfo = new ArrayList<Integer>();
                if (gpuFlag) { // do gpu count
                    if (schemaMap.containsKey(wordID)) {
                        ArrayList<Integer> valueMap = schemaMap.get(wordID);
                        // update the counter
                        for(int j = 0; j < valueMap.size(); j++) {
                            int k = valueMap.get(j);
                            nzw[topic][k] += weight;
                            nz[topic] += weight;
                            gpuInfo.add(k);
                        }
                    } else { // schemaMap don't contain the word

                        // the word doesn't have similar words, the infoMap is empty
                        // we do nothing
                    }
                } else { // the gpuFlag is False
                    // it means we don't do gpu, so the gouInfo map is empty
                }
                wordGPUInfo.get(docID).set(t, gpuInfo); // we update the gpuinfo
                // map
            }
        } else { // we do subtraction according to last iteration information
            for (int t = 0; t < termIDArray.size(); t++) {
                ArrayList<Integer> gpuInfo = wordGPUInfo.get(docID).get(t);
                for(int j = 0; j < gpuInfo.size(); j++) {
                    int simWordId = gpuInfo.get(j);
                    nzw[topic][simWordId] -= weight;
                    nz[topic] -= weight;
                }
            }
        }

    }

    @Override
    public void run() {
        for(int i = index * batch; i < (index + 1) * batch && i < docList.size(); i++) {
            ArrayList<Integer> termIDArray = docList.get(i);
            int preTopic = assignment[i];

            ratioCount(preTopic, i, termIDArray, -1);

            double[] pzDist = new double[numTopic];
            for (int topic = 0; topic < numTopic; topic++) {
                double pz = 1.0 * (mz[topic] + alpha) / (numDoc - 1 + numTopic * alpha);
                double value = 1.0;
//					double logSum = 0.0;
                for (int t = 0; t < termIDArray.size(); t++) {
                    int termID = termIDArray.get(t);
                    value *= (nzw[topic][termID] + beta) / (nz[topic] + vocSize * beta + t);
                    // we do not use log, it is a little slow
                    // logSum += Math.log(1.0 * (nzw[topic][termID] + beta) / (nz[topic] + vocSize * beta + t));
                }
//					value = pz * Math.exp(logSum);
                value = pz * value;
                pzDist[topic] = value;
            }

            for (int k = 1; k < numTopic; k++) {
                pzDist[k] += pzDist[k - 1];
            }
            double u = rg.nextDouble() * pzDist[numTopic - 1];
            int newTopic = -1;
            for (int k = 0; k < numTopic; k++) {
                if (Double.compare(pzDist[k], u) >= 0) {
                    newTopic = k;
                    break;
                }
            }
            // update
            assignment[i] = newTopic;
            ratioCount(newTopic, i, termIDArray, +1);
        }
    }
}
