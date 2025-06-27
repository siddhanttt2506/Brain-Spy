import os
from glob import glob
from nipype.interfaces.fsl import BET, FLIRT
from nipype import logging
import pandas as pd
logging.update_logging(config={"execution": {"remove_unnecessary_outputs": False}})

adni_csv = pd.read_csv("/path/to/csv/file")
output_folder = "/path/to/output_folder"
mni_template = os.environ["FSLDIR"] + "/data/standard/MNI152_T1_1mm.nii.gz"

os.makedirs(output_folder, exist_ok=True)
skullstrip_folder = os.path.join(output_folder, "skullstripped")
registered_folder = os.path.join(output_folder, "registered")
os.makedirs(skullstrip_folder, exist_ok=True)
os.makedirs(registered_folder, exist_ok=True)

nifti_files = adni_csv.path

for nii_path in nifti_files:
    base = os.path.basename(nii_path).replace('.nii.gz', '').replace('.nii', '')
    skull_out = os.path.join(skullstrip_folder, base + "_brain.nii.gz")
    reg_out = os.path.join(registered_folder, base + "_mni.nii.gz")

    bet = BET()
    bet.inputs.in_file = nii_path
    bet.inputs.frac = 0.5
    bet.inputs.robust = True
    bet.inputs.out_file = skull_out
    bet.run()

    flirt = FLIRT()
    flirt.inputs.in_file = skull_out
    flirt.inputs.reference = mni_template
    flirt.inputs.out_file = reg_out
    flirt.inputs.dof = 12
    flirt.run()

print("All images processed.")
