package com.example;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import com.sun.org.apache.xpath.internal.operations.Bool;

public class PGPUDMM {
    public int numTopic;
    public double alpha, beta;
    public int numIter;
    public int roundIndex;
    private Random rg;
    public double weight;
    public int topWords;
    public int filterSize;
    public String similarityFileName;
    public ArrayList<ArrayList<Integer>> topWordIDList;
    public int vocSize;
    public int numDoc;
    public ArrayList<ArrayList<Integer>> docList;
    public double[][] phi;
    private double[] pz;
    private double[][] pdz;
    private double[][] topicProbabilityGivenWord;
    private int numT;

    public ArrayList<ArrayList<Boolean>> wordGPUFlag; // wordGPUFlag.get(doc).get(wordIndex)
    public int[] assignmentList; // topic assignment for every document
    public ArrayList<ArrayList<ArrayList<Integer>>> wordGPUInfo;

    private int[] mz; // have no associatiom with word and similar word
    private double[] nz; // [topic]; nums of words in every topic
    private double[][] nzw; // V_{.k}
    private Map<Integer,ArrayList<Integer>> schemaMap;

    public PGPUDMM(int num_topic, int num_iter, double beta, double alpha, int numT) {
        numTopic = num_topic;
        this.alpha = alpha;
        numIter = num_iter;
        this.beta = beta;
        this.numT = numT;
    }

    /**
     * Collect the similar words Map, not including the word itself
     *
     * @param filename:
     *            shcema_similarity filename
     * @return
     */
    public Map<Integer,ArrayList<Integer>> loadSchema(String filename) {
        Map<Integer, ArrayList<Integer>> schemaMap = new HashMap<Integer, ArrayList<Integer>>();
        try {
            FileInputStream fis = new FileInputStream(filename);
            InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
            BufferedReader reader = new BufferedReader(isr);
            String line;
            int lineIndex = -1;
            while ((line = reader.readLine()) != null) {
                lineIndex++;
                line = line.trim();
                if (line.length() <= 1) {
                    continue;
                }
                String[] items = line.split(" ");
                ArrayList<Integer> tmpArr = new ArrayList<Integer>();
                for (int i = 0; i < items.length; i++) {
                    Integer value = Integer.parseInt(items[i]);
                    tmpArr.add(value);
                }
                if (tmpArr.size() > filterSize || tmpArr.size() == 0) {
                    continue;
                }
                schemaMap.put(lineIndex, tmpArr);
            }
            vocSize = lineIndex + 1;
            return schemaMap;
        } catch (Exception e) {
            System.out.println("Error while reading other file:" + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }


    /**
     * Get the top words under each topic given current Markov status.
     * not used in this RatioGPUDMM
     */
    private ArrayList<ArrayList<Integer>> getTopWordsUnderEachTopic() {
//		compute_pz();
//		compute_phi();
        if (topWordIDList.size() <= numTopic) {
            for (int t = 0; t < numTopic; t++) {
                topWordIDList.add(new ArrayList<Integer>());
            }
        }
        int top_words = topWords;

        for (int t = 0; t < numTopic; ++t) {
            Comparator<Integer> comparator = new TopicalWordComparator(phi[t]);
            PriorityQueue<Integer> pqueue = new PriorityQueue<Integer>(top_words, comparator);

            for (int w = 0; w < vocSize; ++w) {
                if (pqueue.size() < top_words) {
                    pqueue.add(w);
                } else {
                    if (phi[t][w] > phi[t][pqueue.peek()]) {
                        pqueue.poll();
                        pqueue.add(w);
                    }
                }
            }

            ArrayList<Integer> oneTopicTopWords = new ArrayList<Integer>();
            while (!pqueue.isEmpty()) {
                oneTopicTopWords.add(pqueue.poll());
            }
            topWordIDList.set(t, oneTopicTopWords);
        }
        return topWordIDList;
    }

    /**
     * update the p(z|w) for every iteration
     */
    public void updateTopicProbabilityGivenWord() {
        // TODO we should update pz and phi information before
        compute_pz();
        compute_phi();  //update p(w|z)
        for (int i = 0; i < vocSize; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < numTopic; j++) {
                topicProbabilityGivenWord[i][j] = pz[j] * phi[j][i];
                row_sum += topicProbabilityGivenWord[i][j];
            }
            for (int j = 0; j < numTopic; j++) {
                topicProbabilityGivenWord[i][j] = topicProbabilityGivenWord[i][j] / row_sum;  //This is p(z|w)
            }
        }
    }








    public void normalCount(Integer topic, ArrayList<Integer> termIDArray, Integer flag) {
        mz[topic] += flag;
        for (int t = 0; t < termIDArray.size(); t++) {
            int wordID = termIDArray.get(t);
            nzw[topic][wordID] += flag;
            nz[topic] += flag;
        }
    }


    public void initNewModel() {
        schemaMap = loadSchema(similarityFileName);
        wordGPUFlag = new ArrayList<ArrayList<Boolean>>();
        topWordIDList = new ArrayList<ArrayList<Integer>>();
        assignmentList = new int[numDoc];
        wordGPUInfo = new ArrayList<ArrayList<ArrayList<Integer>>>();
        rg = new Random();
        // construct vocabulary
        phi = new double[numTopic][vocSize];
        pz = new double[numTopic];
        pdz = new double[numDoc][numTopic];

//		schema = new double[vocSize][vocSize];
        topicProbabilityGivenWord = new double[vocSize][numTopic];

        for (int i = 0; i < docList.size(); i++) {
            ArrayList<Boolean> docWordGPUFlag = new ArrayList<Boolean>();
            ArrayList<ArrayList<Integer>> docWordGPUInfo = new ArrayList<ArrayList<Integer>>();
            for (int j = 0; j < docList.get(i).size(); j++) {
                docWordGPUFlag.add(false); // initial for False for every word
                docWordGPUInfo.add(new ArrayList<Integer>());
            }
            wordGPUFlag.add(docWordGPUFlag);
            wordGPUInfo.add(docWordGPUInfo);
        }

        // init the counter
        mz = new int[numTopic];
        nz = new double[numTopic];
        nzw = new double[numTopic][vocSize];
    }

    public void init_GSDMM() {
        for (int d = 0; d < docList.size(); d++) {
            ArrayList<Integer> termIDArray = docList.get(d);
            int topic = rg.nextInt(numTopic);
            assignmentList[d] = topic;
            normalCount(topic, termIDArray, +1);
        }

    }
    private static long getCurrTime() {
        return System.currentTimeMillis();
    }

    public void run_iteration() {
        int batch = numDoc / numT + 1;
        workThread[] threads = new workThread[numT];

        for (int iteration = 1; iteration <= numIter; iteration++) {
            System.out.println(iteration + "th iteration begin");
//            for(int i = 0; i < numDoc; i++)
//                System.out.print(assignmentList[i] + " ");
//            System.out.println();
            long _s = getCurrTime();
            updateTopicProbabilityGivenWord();
            for(int i = 0; i < numT; i++) {
                threads[i] = new workThread(i, batch, numTopic, mz, nz, nzw, docList, alpha, beta, weight, numDoc, vocSize, wordGPUInfo, wordGPUFlag, schemaMap, topicProbabilityGivenWord,assignmentList);
                threads[i].start();
            }
            for(int i = 0; i < numT; i++) {
                try {
                    threads[i].join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            // merge
            int[] old_mz = mz.clone();
            double[] old_nz = nz.clone();
            double[][] old_nzw = nzw.clone();
            for(int i = 0; i < numT; i++){
                for(int j = 0; j < numTopic; j++) {
                    mz[j] += threads[i].mz[j] - old_mz[j];
                    nz[j] += threads[i].nz[j] - old_nz[j];
                    for(int v = 0; v < vocSize; v++) {
                        nzw[j][v] += threads[i].nzw[j][v] - old_nzw[j][v];
                    }
                }
            }
            for(int i = 0; i < numDoc; i++){
                int index = i / batch;
                assignmentList[i] = threads[index].assignment[i];
            }
            long _e = getCurrTime();
            System.out.println(iteration + "th iter finished and every iterration costs " + (_e - _s) + "ms "
                    + numTopic + " topics");
        }

    }

    public void saveModel(String flag) {

        compute_phi();
        compute_pz();
        compute_pzd();
        saveTopWords();
        saveModelPz(flag + "_theta.txt");
        saveModelPhi(flag + "_phi.txt");
        saveModelAssign(flag + "_assign.txt");
        saveModelPdz(flag + "_pdz.txt");
    }

    public void saveTopWords() {
        ArrayList<ArrayList<Integer>> TopWordsList = getTopWordsUnderEachTopic();
        try {
            PrintWriter out = new PrintWriter("topWords");
            for (int i = 0; i <  TopWordsList.size(); i++) {
                out.println(i + "th Topic:");
                for (int j : TopWordsList.get(i)) {
                    out.print(j + " ");
                }
                out.println();
            }
            out.flush();
            out.close();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    public void compute_phi() {
        for (int i = 0; i < numTopic; i++) {
            double sum = 0.0;
            for (int j = 0; j < vocSize; j++) {
                sum += nzw[i][j];
            }
            for (int j = 0; j < vocSize; j++) {
                phi[i][j] = (nzw[i][j] + beta) / (sum + vocSize * beta);
            }
        }
    }

    public void compute_pz() {
        double sum = 0.0;
        for (int i = 0; i < numTopic; i++) {
            sum += nz[i];
        }
        for (int i = 0; i < numTopic; i++) {
            pz[i] = 1.0 * (nz[i] + alpha) / (sum + numTopic * alpha);
        }
    }

    public void compute_pzd() {
        double[][] pwz = new double[vocSize][numTopic]; // pwz[word][topic]
        for (int i = 0; i < vocSize; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < numTopic; j++) {
                pwz[i][j] = pz[j] * phi[j][i];
                row_sum += pwz[i][j];
            }
            for (int j = 0; j < numTopic; j++) {
                pwz[i][j] = pwz[i][j] / row_sum;
            }

        }

        for (int i = 0; i < numDoc; i++) {
            ArrayList<Integer> doc_word_id = docList.get(i);
            double row_sum = 0.0;
            for (int j = 0; j < numTopic; j++) {

                for (int wordID : doc_word_id) {
                    pdz[i][j] += pwz[wordID][j];
                }
                row_sum += pdz[i][j];

            }
            for (int j = 0; j < numTopic; j++) {
                pdz[i][j] = pdz[i][j] / row_sum;
            }
        }
    }

    public boolean saveModelAssign(String filename) {
        try {
            PrintWriter out = new PrintWriter(filename);

            for (int i = 0; i < numDoc; i++) {
                int topic = assignmentList[i];
                for (int j = 0; j < numTopic; j++) {
                    int value = -1;
                    if (j == topic) {
                        value = 1;
                    } else {
                        value = 0;
                    }
                    out.print(value + " ");
                }
                out.println();
            }
            out.flush();
            out.close();
        } catch (Exception e) {
            System.out.println("Error while saving assign list: " + e.getMessage());
            e.printStackTrace();
            return false;
        }

        return true;
    }

    public boolean saveModelPdz(String filename) {
        try {
            PrintWriter out = new PrintWriter(filename);

            for (int i = 0; i < numDoc; i++) {
                for (int j = 0; j < numTopic; j++) {
                    out.print(pdz[i][j] + " ");
                }
                out.println();
            }

            out.flush();
            out.close();
        } catch (Exception e) {
            System.out.println("Error while saving p(z|d) distribution:" + e.getMessage());
            e.printStackTrace();
            return false;
        }

        return true;
    }

    public boolean saveModelPz(String filename) {
        // return false;
        try {
            PrintWriter out = new PrintWriter(filename);

            for (int i = 0; i < numTopic; i++) {
                out.print(pz[i] + " ");
            }
            out.println();

            out.flush();
            out.close();
        } catch (Exception e) {
            System.out.println("Error while saving pz distribution:" + e.getMessage());
            e.printStackTrace();
            return false;
        }

        return true;
    }

    public boolean saveModelPhi(String filename) {
        try {
            PrintWriter out = new PrintWriter(filename);

            for (int i = 0; i < numTopic; i++) {
                for (int j = 0; j < vocSize; j++) {
                    out.print(phi[i][j] + " ");
                }
                out.println();
            }
            out.flush();
            out.close();
        } catch (Exception e) {
            System.out.println("Error while saving word-topic distribution:" + e.getMessage());
            e.printStackTrace();
            return false;
        }

        return true;
    }


    public void LoadCorpus(String filename) {
        docList = new ArrayList<ArrayList<Integer>>();
        try {
            FileInputStream fis = new FileInputStream(filename);
            InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
            BufferedReader reader = new BufferedReader(isr);
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                String[] items = line.split(" ");
                ArrayList<Integer> tmpArr = new ArrayList<Integer>();
                for (int i = 0; i < items.length; i++) {
                    Integer value = Integer.parseInt(items[i]);
                    tmpArr.add(value);
                }
                docList.add(tmpArr);
            }
            numDoc = docList.size();
        } catch (Exception e) {
            System.out.println(docList.size());
            System.out.println("Error while reading other file:" + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        int num_iter = 1000;
        double beta = 0.1;
        String similarityFileName = "sim.txt";
        double weight = 0.1;
        int filterSize = 20;
        int numT = 4;
        int num_topic = 50;
        double alpha = 1.0 * 50 / num_topic;
        PGPUDMM model = new PGPUDMM(num_topic, num_iter, beta, alpha, numT);
        model.LoadCorpus("weibo.txt");
        model.filterSize = filterSize;
        model.roundIndex = 1;
        model.topWords = 15;
        model.similarityFileName = similarityFileName;
        model.weight = weight;
        model.initNewModel();
        model.init_GSDMM();
        model.run_iteration();
        model.saveModel("PGPUDMM ");
    }
}



/**
 * Comparator to rank the words according to their probabilities.
 */
class TopicalWordComparator implements Comparator<Integer> {
    private double[] distribution = null;

    public TopicalWordComparator(double[] distribution2) {
        distribution = distribution2;
    }

    @Override
    public int compare(Integer w1, Integer w2) {
        if (distribution[w1] < distribution[w2]) {
            return -1;
        } else if (distribution[w1] > distribution[w2]) {
            return 1;
        }
        return 0;
    }
}
