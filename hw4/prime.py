def is_prime(n):
    if n < 2:
        return "EXCEPTION"
    if n == 2:
        return "PRIME 1"
    
    analysis_count = 0
    for i in range(2, int(n**0.5) + 1):
        analysis_count += 1
        if n % i == 0:
            return "COMPOSITE"
    
    return f"PRIME {analysis_count}"

def main(input_file):
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            if line.isdigit():
                number = int(line)
                result = is_prime(number)
            else:
                result = "EXCEPTION"
            print(result)
    except FileNotFoundError:
        print("error ^^ ~please input a file.")
    except Exception as e:
        print(f"{e}")

if __name__ == "__main__":
    input_file = 'input.txt'
    main(input_file)
