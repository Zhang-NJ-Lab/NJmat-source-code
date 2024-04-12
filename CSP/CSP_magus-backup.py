import subprocess
# 多条命令
command1 = "magus -v"
command2 = "magus generate -i input.yaml -o gen.traj -n 10 -s -o "+"outputdir_path"
command3 = "magus summary gen.traj -s"

# 将多条命令串联到一个字符串中
magus_command = f"{command1} & {command2} & {command3}"
# Magus命令
# magus_command = "C:/Users/Jarvis/miniconda3/envs/myenv/Scripts/magus -v"

# 使用subprocess模块运行Magus命令
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
