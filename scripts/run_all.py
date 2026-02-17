
import subprocess
import concurrent.futures

checkpoints = ["trans_long-8", "trans_long-11"]
results_file = "all_checkpoints_results.txt"

def evaluate(ckpt):
    print(f"Starting evaluation for {ckpt}...")
    try:
        # Run the command and capture output
        # Using shell=True to ensure python command is found in path
        # Updated to point to evaluation/test_ascad.py
        result = subprocess.run(f"python evaluation/test_ascad.py {ckpt}", shell=True, capture_output=True, text=True)
        
        output = result.stdout
        error = result.stderr
        
        return ckpt, output, error, result.returncode
    except Exception as e:
        return ckpt, "", str(e), -1

print(f"Evaluating {len(checkpoints)} checkpoints sequentially...")

with open(results_file, "w", encoding="utf-8") as f:
    for ckpt in checkpoints:
        ckpt_name, out, err, code = evaluate(ckpt)
        
        f.write(f"========================================\n")
        f.write(f"CHECKPOINT: {ckpt_name}\n")
        f.write(f"RETURN CODE: {code}\n")
        f.write(f"========================================\n")
        f.write("STDOUT:\n")
        f.write(out + "\n")
        f.write("STDERR:\n")
        f.write(err + "\n")
        f.write("\n\n")
        print(f"Finished {ckpt}. Return code: {code}")

print(f"All done! Results saved to {results_file}")
