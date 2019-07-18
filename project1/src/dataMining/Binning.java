package dataMining;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;

public class Binning {
	public static DiscretizationInfo  generateBins(List<Double> objList,String strGeneNo){
		String binInfo ="";
		DiscretizationInfo objInfo = new DiscretizationInfo();
		if(objList != null){
			Collections.sort(objList);
			double Lb = objList.get(0);
			double Ub = objList.get(objList.size()-1);
			int noBin = objList.size()/3; 
			double binMax1 = ((objList.get(noBin)+objList.get(noBin+1))/2);
			double binMax2 = ((objList.get(noBin+noBin+1)+objList.get(noBin+noBin+2))/2);
			int count1 = 0;
			int count2 = 0;
			int count3 =0;
			objInfo.binMax1 = binMax1;
			objInfo.binMax2 = binMax2;
			for(int i = 0; i< objList.size(); i++){
				if(objList.get(i)<=binMax1){
					count1++;
				}else if(objList.get(i)<=binMax2){
					count2++;
				}else{
					count3++;
				}
				
			}
			Double variance = getVariance(objList);
			 binInfo = strGeneNo+": Variance :"+roundVal(variance,4) +", Bins : " +"[-inf ,"+roundVal(binMax1,4)+"],"+count1+
					 ",("+roundVal(binMax1,4)+(",")+roundVal(binMax2,4)+"],"+count2+",("+roundVal(binMax2,4)+",+inf],"+count3;
			// System.out.println(binInfo);
			 objInfo.binInfoData = binInfo;
			
		}
		
		return objInfo;
		
	}
	//function to calculate variance
	public static Double getVariance(List<Double> objList){
		double variance =0;
		double mean =0;
		double sum =0;
		double sumSquare = 0;
		if((!objList.isEmpty()) &&(objList.size()>0)){
		for(Double item:objList){
			sum = sum+item;
		}
		mean = sum/objList.size();
		
		for (Double item : objList){
			sumSquare =sumSquare +(item - mean)*(item - mean);
			
		}
			
		variance = sumSquare/(objList.size());
			
		}
		
		return variance;
		
	}
	public static double roundVal(double value, int places) {
	     if (places < 0) throw new IllegalArgumentException();

	     BigDecimal bd = new BigDecimal(value);
	     bd = bd.setScale(places, RoundingMode.HALF_UP);
	     return bd.doubleValue();
	    }
	public void getBinsInfo(String fileName,int K) throws Exception{
		int i =0;
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		Double[] objClass =null;
		List<Double> obClassList = new ArrayList<Double>();
		LinkedHashMap<String,List<Double>> objHashMap = new LinkedHashMap<String,List<Double>>();
		for(String line; (line = br.readLine()) != null; ) {
			String [] strSplit = line.split(",");
					
	       for(int j=0;j<K;j++){
	    	   if(i==0){
	    		   List<Double> objList = new ArrayList();
	    		   objList.add(Double.parseDouble(strSplit[j]));
	    		   objHashMap.put("G"+(j+1), objList);
				}else{
					objHashMap.get("G"+(j+1)).add(Double.parseDouble(strSplit[j]));
					
					
				}
	    	   
	       }
	       
	    	   if(strSplit[strSplit.length -1].equals("positive")){
	    		   obClassList.add(1.0); 
	    	   }else{
	    		   obClassList.add(0.0); 
	    	   }
	    	   
	       
	       
	       i=i+1;
	    }
		objClass = obClassList.toArray(new Double[obClassList.size()]);
		
		
		Iterator<String> objIter = objHashMap.keySet().iterator();
		//System.out.println("total size:" +objHashMap.size());
		
		BufferedWriter out1 = new BufferedWriter(new FileWriter("eDensityBins.txt"));
		List<List<Double>> objBinValues = new ArrayList<List<Double>>();
		while(objIter.hasNext()){
			String objKey = objIter.next();
			List<Double> objList = objHashMap.get(objKey);
			List<Double> newList = new ArrayList<Double>();
			List<Double> binValue = new ArrayList<Double>();
		    for (Double item : objList) { newList.add(item); }
		    DiscretizationInfo objInfo =generateBins(newList,objKey);
		    binValue.add(objInfo.binMax1);
		    binValue.add(objInfo.binMax2);
		    objBinValues.add(binValue);
		    out1.write(objInfo.binInfoData);
		    out1.write("\n");    
		}
		out1.close();
		BufferedReader br1 = new BufferedReader(new FileReader(fileName));
		BufferedWriter out = new BufferedWriter(new FileWriter("eDensityData.txt"));
		for(String line; (line = br1.readLine()) != null; ) {
			String [] strSplit = line.split(",");
			for (int b =0; b<K; b++){
				if(Float.parseFloat(strSplit[b])<= objBinValues.get(b).get(0)){
					out.write("a,");
				}else if(Float.parseFloat(strSplit[b])<= objBinValues.get(b).get(1)){
					out.write("b,");
				}else{
					out.write("c,");
				}
				
			}
			out.write(strSplit[strSplit.length-1]);
			out.write("\n");
		}
		out.close();
		
		System.out.println("success!!!");
	}
	
}

