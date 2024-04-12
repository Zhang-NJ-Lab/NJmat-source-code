import subprocess

def run_magus_command(inpath,outpath):
    # 多条命令
    print(inpath)
    print(outpath)
    path=outpath+"/gen.traj"

    inpath = inpath.replace("/", "\\")
    outpath = outpath.replace("/", "\\")
    path = path.replace("/", "\\")
    print(inpath)
    command1 = "magus -v"
    #command2 = "magus generate -i C:\\Users\\DELL\\Desktop\\input.yaml -o gen.traj -n 20"
    command2 = "magus generate -i "+inpath+" -o "+path+" -n 20"
    command3 = "magus summary "+path+" -s -o "+outpath
    print(command2)
    print(command3)

    # 将多条命令串联到一个字符串中
    magus_command = f"{command1} & {command2} & {command3}"

    try:
        process = subprocess.Popen(magus_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return_code = process.returncode
        if return_code == 0:
            print("Magus command executed successfully")
            print("Output:", stdout.decode())
        else:
            print("Magus command failed")
            print("Error:", stderr.decode())
    except Exception as e:
        print("An error occurred:", e)