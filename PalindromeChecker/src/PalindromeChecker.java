import java.util.Scanner;
public class PalindromeChecker {
 public static boolean isPalindrome(String str){
       
    
  if(str==null || str.length() == 0 || str.length() == 1){
            return true;
     }
        else{
            if(str.charAt(0) == str.charAt(str.length()-1)){
                return isPalindrome(str.substring(1,str.length()-1));
         }else{
                return false;
            		}
        }
}

    public static void main(String args[])	{
     Scanner scanner = new Scanner(System.in);
      System.out.print("Enter a string: ");
     String str = scanner.nextLine();

   if (isPalindrome(str)) {
            System.out.println(str + " is palindrome");
        
   } else {
            System.out.println(str + " is not palindrome");
}
   scanner.close();
    }       
}