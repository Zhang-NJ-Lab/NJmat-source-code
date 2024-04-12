import paramiko
import sys
from Visualizer.demo import Ui_AseAtomInput
from PyQt5.QtWidgets import *
from ase.visualize import view
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp
from ase.io import read

### 创建MainWindow
class ASE_ui(Ui_AseAtomInput,QMainWindow): 
    def __init__(self,) -> None:
        super().__init__()
        self.atom_calculator = AtomCalculator()
        self.initUI()


    def initUI(self):
        self.setupUi(self)
        self.retranslateUi(self)


        self.DelAtom.clicked.connect(self.deleteAtom)
        self.AddAtom_Buttom.clicked.connect(self.addAtom)
        self.ViewStructure_Button.clicked.connect(self.showStructure)
        self.SetCaluator_Button.clicked.connect(self.setCalculator)
        self.actionFrom.triggered.connect(self.importCIF)
        self.actionServer.triggered.connect(self.connectRemoteServer)
        self.Process_Button.clicked.connect(self.startCalculation)

    def addAtom(self):
        element = self.AtomInPut.text()
        x = float(self.X_Input.text())
        y = float(self.Y_Input.text())
        z = float(self.Z_Input.text())

        self.atom_calculator.addAtom(element, x, y, z)

        self.AtomInPut.clear()
        self.X_Input.clear()
        self.X_Input.clear()
        self.X_Input.clear()

    ### 对应：“删除上一个原子”按钮
    def deleteAtom(self):
        self.atom_calculator.deleteAtom()

    ### 对应：“结束输入”
    def finishInput(self):
        self.close()

    ### 对应：“显示原子结构”
    def showStructure(self):
        if self.atom_calculator.atoms is not None:
            view(self.atom_calculator.atoms)

    ### 对应：“设置计算器”按钮
    def setCalculator(self):
        calculator_name = self.Caluator_Input.text()
        self.atom_calculator.setCalculator(calculator_name)
        if calculator_name.lower() == "vasp":
            self.showVaspParamDialog()
        else:
            show_Calculator = QErrorMessage(self)
            show_Calculator.showMessage(f"caulater: {calculator_name}")

    ### 点击“设置计算器”按钮之后如果是“VASP”计算器就显示输入计算参数
    def showVaspParamDialog(self):
        self.dialog = VaspParamDialog(self)
        if self.dialog.exec_() == QDialog.Accepted:
            self.vasp_params = self.dialog.getParams()
            show_Calculator = QErrorMessage(self)
            show_Calculator.showMessage(f"Caulater: vasp, paragram: {self.vasp_params}")


    def setOptimizer(self):
        optimizer_name = self.optimizer_input.text()
        self.atom_calculator.setOptimizer(optimizer_name)
        show_Optimizer = QErrorMessage(self)
        show_Optimizer.showMessage(f"优化器为: {optimizer_name}")

    ### 对应：“开始计算”按钮
    def startCalculation(self):
        self.atom_calculator.startCalculation()
        energy = self.atom_calculator.energy
        forces = self.atom_calculator.forces
        message = QErrorMessage(self)
        message.showMessage(f"energy: {energy}, force为: {forces}")

    def importCIF(self):
        dialog = CIFImportDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            atoms = dialog.getAtoms()
            if atoms is not None:
                self.atom_calculator.atoms = atoms
                show_Structure = QErrorMessage(self)
                show_Structure.showMessage('Successfully Improt Atom Info')
    
    def connectRemoteServer(self):
        dialog = RemoteServerDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            server_info = dialog.getServerInfo()
            if server_info is not None:
                hostname = server_info['hostname']
                username = server_info['username']
                password = server_info['password']


                try:
                    ssh_client = paramiko.SSHClient()
                    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh_client.connect(hostname=hostname, username=username, password=password)

                    command = "vasp_std"  # 根据远程服务器上的VASP命令调整命令名称
                    vasp_calculator = Vasp(command=command)

                    # 设置VASP计算器
                    self.atom_calculator.setCalculator('vasp')
                    self.atom_calculator.vasp_parameters = server_info
                    self.atom_calculator.vasp_calculator = vasp_calculator

                    QMessageBox.information(self, f'successful', f'successfully connect to {hostname}')

                except Exception as e:
                    error_dialog = QErrorMessage(self)
                    error_dialog.showMessage(f'connect faild：{str(e)}')

# 链接远程服务器类，用于获取远程服务器的连接信息
class RemoteServerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('connect Remote Server')
        self.setModal(True)
        self.server_info = None

        self.hostname_label = QLabel('hosthame:')
        self.username_label = QLabel('acount:')
        self.password_label = QLabel('passwd:')

        self.hostname_input = QLineEdit()
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.hostname_label)
        layout.addWidget(self.hostname_input)
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def getServerInfo(self):
        self.server_info = {
            'hostname': self.hostname_input.text(),
            'username': self.username_input.text(),
            'password': self.password_input.text(),
        }
        return self.server_info
### 从CIF文件导入类
class CIFImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('From CIF')
        self.setModal(True)
        self.filepath = None

        self.filepath_label = QLabel('CIF Path:')
        self.filepath_input = QLineEdit()
        self.btn_browse = QPushButton('browse', self)
        self.btn_browse.clicked.connect(self.browseCIF)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.filepath_label)
        layout.addWidget(self.filepath_input)
        layout.addWidget(self.btn_browse)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def browseCIF(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter('CIF files (*.cif)')
        if file_dialog.exec_():
            self.filepath_input.setText(file_dialog.selectedFiles()[0])

    def getAtoms(self):
        self.filepath = self.filepath_input.text()
        try:
            atoms = read(self.filepath)
            return atoms
        except Exception as e:
            error_dialog = QErrorMessage(self)
            error_dialog.showMessage(f'Could not improt CIF file：{str(e)}')
            return None

### VASP 设置参数类
class VaspParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('set VASP parameter')
        self.setModal(True)

        self.xc_label = QLabel('xc:')
        self.xc_input = QLineEdit()

        self.kpts_label = QLabel('kpts:')
        self.kpts_input = QLineEdit()

        self.encut_label = QLabel('encut:')
        self.encut_input = QLineEdit()

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.xc_label)
        layout.addWidget(self.xc_input)
        layout.addWidget(self.kpts_label)
        layout.addWidget(self.kpts_input)
        layout.addWidget(self.encut_label)
        layout.addWidget(self.encut_input)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def getParams(self):
        xc = self.xc_input.text()
        kpts = eval(self.kpts_input.text())
        encut = int(self.encut_input.text())
        print()
        return {'xc': xc, 'kpts': kpts, 'encut': encut , 'command':f'""'} 

### 内核主控制模块
class AtomCalculator:
    def __init__(self):
        self.atoms_list = []
        self.atoms = None
        self.calculator_name = None
        self.optimizer_name = None
        self.energy = None
        self.forces = None

    def addAtom(self, element, x, y, z):
        atom_info = {'symbol': element, 'position': (x, y, z)}
        self.atoms_list.append(atom_info)
        self.updateStructure()

    def deleteAtom(self):
        if self.atoms_list:
            self.atoms_list.pop()
            self.updateStructure()

    def updateStructure(self):
        symbols = [atom['symbol'] for atom in self.atoms_list]
        positions = [atom['position'] for atom in self.atoms_list]
        self.atoms = Atoms(symbols=symbols, positions=positions)

    def setCalculator(self, calculator_name):
        self.calculator_name = calculator_name

    def setOptimizer(self, optimizer_name):
        self.optimizer_name = optimizer_name

    def startCalculation(self):
        if self.atoms is not None:
            if self.calculator_name == "emt":
                emt = EMT()
                self.atoms.set_calculator(emt)
                self.energy = self.atoms.get_potential_energy()
                self.forces = self.atoms.get_forces()

            elif self.calculator_name == "vasp":
                vasp_params = {
                    'xc': 'PBE',
                    'kpts': (4, 4, 4),
                    'encut': 400,
                }

                vasp_calculator = Vasp(txt = "vasp.out",**vasp_params)
                self.atoms.set_calculator(vasp_calculator)

                self.energy = self.atoms.get_potential_energy()
                self.forces = self.atoms.get_forces()

            elif self.calculator_name == "gaussian":
                pass

            else:
                self.energy = None
                self.forces = None

        else:
            self.energy = None
            self.forces = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ASE_ui()
    window.show()
    sys.exit(app.exec_())

    