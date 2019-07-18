package dataMining;

import static java.lang.Math.log;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.RandomAccessFile;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class EntropyCalculation {
	//this function calculates highest info gain 
	public static  double[] calculateInfoGain(Double[] objGeneData, Double[] objClassData, double objEntropyAll){
        double topGain= 0, maxSplitVal=0, maxIndx=0; 
        for (int i=0 ; i< objGeneData.length-1; i++){
            int s1Val=0, s2Val=0;
            double splitVal= (objGeneData[i]+objGeneData[i+1]) /2;
            for (Double geneData1 : objGeneData) {
                if (geneData1 < splitVal) {
                    s1Val++;
                } else {
                    s2Val++;
                }
            }
            double pVal1= 0,nVal1=0;
            double pVal2=0,nVal2=0;
            for (int a=0; a<s1Val;a++){
               if (objClassData[a]==1){
                   pVal1++;
               }
               else
                   nVal1++;
            }
            for (int b=s1Val; b<s2Val+s1Val;b++){
               if (objClassData[b]==1){
                   pVal2++;
               }
               else
                   nVal2++;
            }
            float infoPs1 = (float) (log(pVal1/s1Val)/log(2));
            if (Double.isInfinite(infoPs1)) { infoPs1 = 0;}
            
            float infoNs1 = (float) (log(nVal1/s1Val)/log(2));
            if (Double.isInfinite(infoNs1)) { infoNs1 = 0; }
            
            float infoPs2 = (float) (log(pVal2/s2Val)/log(2));
            if (Double.isInfinite(infoPs2)) { infoPs2 = 0;}
            
            float infoNs2 = (float) (log(nVal2/s2Val)/log(2));
            if (Double.isInfinite(infoNs2)) { infoNs2 = 0; }
            
            double entropyS1 =-(pVal1/s1Val)*(infoPs1)-(nVal1/s1Val)*(infoNs1);
            double entropyS2 =-(pVal2/s2Val)*(infoPs2)-(nVal2/s2Val)*(infoNs2);
            double rS1 = (double)s1Val/(s1Val+s2Val);
            double rS2 = (double)s2Val/(s1Val+s2Val);
            double information =entropyS1*(rS1)+ entropyS2*(rS2);
            double tempGain =objEntropyAll - information;
            if (topGain<tempGain){
                topGain = tempGain;
                maxIndx = i;
                maxSplitVal= splitVal;
            }
            
        }
        //System.out.println("max gain:"+maxGain);
        double[] maxInfo = new double[3];
        maxInfo[0] = topGain;
        maxInfo[1]= maxIndx;
        maxInfo[2]= maxSplitVal;
        return maxInfo;
        
}
	public static double entropyOfAll(Double[] objClassData){
		float nCount = 0;
		float pCount = 0;
		for(int i = 0; i<objClassData.length; i++){
			if (objClassData[i]== 0){
				nCount+=1; 
			}
			else
				pCount+=1;
		}
		double p=pCount/(nCount+pCount);
		double n=nCount/(nCount+pCount);
		float pS = (float) (log(p)/log(2));
		float nS = (float) (log(n)/log(2));
		if (Double.isInfinite(pS)) {pS = 0;}
		if (Double.isInfinite(nS)) {nS = 0;}
		
		double entropyAll1 =(double)(-p*(pS)-n*(nS));
		return entropyAll1;
	}
	public static double roundVal(double value, int places) {
		if (places < 0) throw new IllegalArgumentException();

		BigDecimal bd = new BigDecimal(value);
		bd = bd.setScale(places, RoundingMode.HALF_UP);
		return bd.doubleValue();
	}

	public static void main(String[] args) throws Exception { 
		Long startTime = System.currentTimeMillis();
		FileWriter writer = null;
		//fileName ="p1colon.txt";
		String strFileName = args[0];//input file
		int k =200;
		System.out.println("value of K is taken 200 by default");
		System.out.println("task 1 starts here");
		Binning objBinning = new Binning();
		//function to call firts task
		objBinning.getBinsInfo(strFileName, k);
		System.out.println("task 1 ends here");
		System.out.println("output written to file eDensityBins.txt and eDensityData.txt");
		//output files for task2 and task 3
		BufferedWriter brBins = new BufferedWriter(new FileWriter("entropyBins.txt"));
		BufferedWriter breData = new BufferedWriter(new FileWriter("entropyData.txt"));
		BufferedWriter brCorr = new BufferedWriter(new FileWriter("correlatedGens.txt"));
		File file = new File(strFileName);
		Scanner objScanner = new Scanner(file);
		File file2 = new File(strFileName+".csv");
		file2.createNewFile();
		writer = new FileWriter(file2);

		while (objScanner.hasNext()) {
			String objScan = objScanner.nextLine();
			writer.append(objScan);
			writer.append("\n");
			writer.flush();
		}

		DataSource frData1;  
		frData1 = new DataSource( strFileName+".csv" );
		Instances instance2 = frData1.getDataSet();
		String header = "g1";
		for( int i=2;i<=instance2.numAttributes()-1; i++){
			header = header+", g"+i; 
		}
		header = header+", class";

		RandomAccessFile file1 = new RandomAccessFile(strFileName+".csv", "rws");
		byte[] text = new byte[(int) file1.length()];
		file1.readFully(text);
		file1.seek(0);
		file1.writeBytes(header);
		file1.writeBytes("\n");
		file1.write(text);
		file1.close();
		//System.out.println("Done");

		DataSource objDataSource;  
		objDataSource = new DataSource( strFileName+".csv" );
		Instances instance1 = objDataSource.getDataSet();

		instance1.setClassIndex( instance1.numAttributes() - 1 );
		Double[] objGeneData = new Double[instance1.numInstances()];
		Double[] objClassData = new Double[instance1.numInstances()];
		AttributeSelection filter = new AttributeSelection();     
		Ranker objRanker= new Ranker();//Weka search method
		OneRAttributeEval objAttrEval = new OneRAttributeEval();//Weka evaluation method
		filter.setEvaluator(objAttrEval);  
		filter.setSearch(objRanker);  
		filter.setInputFormat(instance1);   
		Instances instance3 = Filter.useFilter( instance1, filter); 
		instance3.setClassIndex(instance3.numAttributes() - 1);
		String[] task3temp= new String[k];

		for (int j=0; j<k; j++)
		{
			task3temp[j]= instance3.attribute(j).name();
		}

		//Task 2 Begins
		for(int j = 0;j<instance1.numInstances(); j++ ){
			objClassData[j] = instance1.instance(j).value(instance1.numAttributes()-1);
		}
		
		String[] objGeneNames = new String[instance1.numAttributes()-1];
		Double[] objTopGain = new Double[instance1.numAttributes()-1];
		double entropyAll = entropyOfAll(objClassData);//calling function to calcuate entropy of values for a single gene
		for (int j=0; j<instance1.numAttributes()-1; j++){
			for (int i=0; i<instance1.numInstances(); i++){
				objGeneData[i] = instance1.instance(i).value(j);
			}
			Map<Double, Double> objGeneDataMap = new HashMap<Double, Double>() {};
			for(int indx=0; indx < objGeneData.length; indx++){
				objGeneDataMap.put(Double.valueOf(objGeneData[indx]), Double.valueOf(objClassData[indx]));
			}
			Arrays.sort(objGeneData);

			for(int indx=0; indx < objGeneData.length; indx++){
				objClassData[indx] = objGeneDataMap.get(objGeneData[indx]).doubleValue();
			}
			double[] retSp = new double[3];
			retSp = calculateInfoGain(objGeneData, objClassData, entropyAll);//method called to find the highest information gain
			int indxx  = (int) retSp[1];
			int pBin1=0,nBin1=0,pBin2=0,nBin2=0;
			for (int i=0;i<=indxx;i++){
				if(objClassData[i]==1){
					pBin1++;
				} 
				else
					nBin1++;

			}
			for (int i=(int)(indxx+1); i<objClassData.length;i++){
				if(objClassData[i]==1){
					pBin2++;
				} 
				else
					nBin2++;
			}
			//System.out.println("maxgain: "+ retSp[0]);
			int bin1Total = pBin1+nBin1;
			int bin2Total = pBin2+nBin2;

			objGeneNames[j] = instance1.attribute(j).name()+","+retSp[2]+","+bin1Total+","+bin2Total;
			objTopGain[j] = retSp[0];


		}
		Map<Double, String> objTopGainToGeneMap = new HashMap<Double, String>() {};
		for(int objIndx=0; objIndx < objTopGain.length; objIndx++){
			objTopGainToGeneMap.put(Double.valueOf(objTopGain[objIndx]), String.valueOf(objGeneNames[objIndx]));
		}
		Arrays.sort(objTopGain);
		for(int objIndx=0; objIndx < objTopGain.length; objIndx++){
			objGeneNames[objIndx] = objTopGainToGeneMap.get(objTopGain[objIndx]).toString();
		}
		int num=k;
		int len=objTopGain.length-1;
		String[] task2output= new String[num];
		int var=0;
		//Block to produce entropybins.txt
		while(num!=0){
			String[] allVal = objGeneNames[len].split(",");
			task2output[var] = allVal[0];
			brBins.write(allVal[0]+":  Info Gain: "+roundVal(objTopGain[len],4)+"; Bins:"
					+ " (-, "+roundVal(Double.parseDouble(allVal[1]),4)+"] , "+allVal[2]+"; ("+roundVal(Double.parseDouble(allVal[1]),4)+", "
					+ "+] ,"+allVal[3]+"\n");
			len--;
			num--;
			var++;
		}
		//Block to produce entropydata.txt begins
		int j=0;
		while(j<instance1.numInstances()){
			int num1=k;
			int x=objTopGain.length-1;
			while(num1!=0){ 
				String[] val = objGeneNames[x].split(",");
				for (int i=0;i<instance1.numAttributes()-1;i++)
				{
					double val1= Double.parseDouble(val[1]);
					double val2= Double.parseDouble(val[2]);
					if(instance1.attribute(i).name().equals(val[0]))
					{
						if(val1>instance1.instance(j).value(i)){
							breData.write("a, ");
						}else{
							breData.write("b, ");
						}
					}

				}
				x--;
				num1--;
			}
			int dataMapped= (int)instance1.instance(j).value(instance1.numAttributes()-1);
			String objClass = "";
			if (dataMapped ==0)
				objClass = "negative";
			else
				objClass= "positive";     
			breData.write(objClass);
			breData.write("\n");
			j++;
		}
		System.out.println("Completion of task 2.");
		System.out.println("output written to file entropyBins.txt and entropyData.txt");
		
		
		//starting of task 3
		String[] objValues=new String[k*k];
		Double[] correlationVal=new Double[k*k];
		int z=0,y=0; 
		for (int m=0;m<task3temp.length;m++){
			++y;
			for (int n=0;n<task2output.length;n++){
				String gene1 = task3temp[m];
				String gene2 = task2output[n];
				Double[] valGene1 = new Double[instance1.numInstances()];
				Double[] valGene2 = new Double[instance1.numInstances()];
				for(int p =0; p<instance1.numAttributes()-1;p++){

					if(instance1.attribute(p).name().equals(gene1)){
						for(int q =0;q<instance1.numInstances();q++){
							valGene1[q] = instance1.instance(q).value(p);
						}
					}
					if(instance1.attribute(p).name().equals(gene2)){
						for(int q =0;q<instance1.numInstances();q++){
							valGene2[q] = instance1.instance(q).value(p);
						}
					}
				}
				double mean1=0,total1=0,total2=0,mean2=0,sDev1=0,sDev2=0;
				double d1=0,d2=0,covariance=0,correlation=0;
				for(int c=0; c<valGene1.length;c++){
					total1+=valGene1[c];
					total2+=valGene2[c];
				}
				mean1 = total1/valGene1.length; //calculating means
				mean2 = total2/valGene2.length;
				for(int d=0;d<valGene1.length;d++){
					d1+=(valGene1[d]-mean1)*(valGene1[d]-mean1);
					d2+=(valGene2[d]-mean2)*(valGene2[d]-mean2);
				}
				sDev1=Math.sqrt(d1/valGene1.length);//calculating standard deviations
				sDev2=Math.sqrt(d2/valGene2.length);
				for (int e=0;e<valGene1.length;e++){
					covariance+= (valGene1[e]-mean1)*(valGene2[e]-mean2); 
				}
				covariance=covariance/valGene1.length;
				correlation= covariance/(sDev1*sDev2);
				// pairs[z] = "("+trim(gene1)+", "+trim(gene2)+") :";
				objValues[z] = "("+(gene1)+", "+(gene2)+") :";
				correlationVal[z] = correlation+0.0000001*y;
				z++;
			}
		}

		Map<Double, String> objHashMap = new HashMap<Double, String>() {};
		for(int indx=0; indx < correlationVal.length; indx++){
			objHashMap.put(Double.valueOf(correlationVal[indx]), String.valueOf(objValues[indx]));
		}
		Arrays.sort(correlationVal);
		//Writing correlation in file in descending order
		for(int indx=0; indx < correlationVal.length; indx++){
			objValues[indx] = objHashMap.get(correlationVal[indx]).toString();
		}
		int n=k*k;
		int l=correlationVal.length-1;
		while(n!=0){
			brCorr.write(objValues[l]+roundVal(correlationVal[l],4)+"\n");
			l--;
			n--;
		}

		System.out.println("completion of task 3.......");
		System.out.println("output written to file correlatedGens.txt");
		brBins.close();
		breData.close();
		brCorr.close();
		System.out.println("All Task Completed Successfully!!!");
		Long endTime = System.currentTimeMillis();
        System.out.println("Time for completion of whole task in second: "+(endTime - startTime)*0.001);
	}

}
