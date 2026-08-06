import os
import subprocess

if __name__ == "__main__":
    files = [f for f in os.listdir() if f.endswith(".pdb")]
    answer = "9\n1\n"
    for f in files:
        name = f.split(".")[0].lower().replace(" ", "_")
        # subprocess.run(["obabel", f, "-O", f"{name}.pdb"])
        subprocess.run(
            ["gmx_mpi", "pdb2gmx", "-f", f"{name}.pdb", "-o", f"{name}.gro", "-p", f"{name}.top", "-ignh"],
            input=answer,
            text=True)
