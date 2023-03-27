package wilcox_avery_hw2_wvpp;
import java.io.FileNotFoundException;
import java.io.*;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;
public class wilcox_avery_hw2_wvpp {
	// Avery Wilcox
	// 11/11/20
	// Computer Science 110 Section 20
	// Homework #2: West Virginia Population Problem

	//Creates method for calculating Population Desnity
	public static double popDenstiy(double pop, double area) {
return (pop / area);
	}
	// My attempt to write the Hash Map to a file
//public static HashMap fileWrite(String a, double b) {
	
	// Iterate over HashMap
	
	//try (var writer = new FileWriter("popdensity.csv")){
		//var element = hashMap.entrySet();
		//writer.write(element);
		// Iterate over HashMap
			//System.out.printf("%s: %s%n", element.getKey(), element.getValue());

	
	public static void main(String[] args) {
		// Opening the file from counties.csv
		try (var bufferedReader = new BufferedReader(new FileReader("counties.csv"))){
		//Throw away the first line	
			bufferedReader.readLine();
		
			
			
			//Loop to process the column
			String line;
			while((line = bufferedReader.readLine()) != null) {
				// Get line columns
				var columns = line.split(",");
				//Prepare County
				var county = columns[0].strip();
			// Prepare Population
				var pop = Integer.parseInt(columns[4].strip());
			//Prepare Area
				var area = Integer.parseInt(columns[5].strip());
						
						//Creates Hash Map to store information
						var hashMap = new HashMap<String, Double>();
						//Adding elements to the Hash Map
						hashMap.put(county,wilcox_avery_hw2_wvpp.popDenstiy(pop, area));
						// Remove Upshur County
						hashMap.remove("Upshur");
						//Remove Tyler County
						hashMap.remove("Tyler");
						//Not necessary but used to show whats in the hash map
						System.out.println(hashMap);
						// wilcox_avery_hw2_wvpp.fileWrite(hashMap, null);
			}
			
			
			// Tells the user if the file can not be found
		}catch (FileNotFoundException e) {
			System.out.println("File could not be located");
		//Tells the user is there is an IO error
		} catch (IOException e) {
			
			System.out.println("IO Exception");
			}
			
	}



}
