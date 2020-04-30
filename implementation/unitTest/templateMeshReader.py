from templateManager.templateMesh import get_template_mesh
from helper.shapenet.shapenetMapper import desc_to_id
import numpy as np

id = desc_to_id("pistol")
print(id)
mesh = get_template_mesh("/media/saurabh/e56e40fb-030d-4f7f-9e63-42ed5f7f6c71/preprocessing", id)
with mesh:
    a, b = mesh['2137b954f778282ac24d00518a3dd6ec']['faces'], mesh['2137b954f778282ac24d00518a3dd6ec']['vertices']
    print(np.array(a))
    print(np.array(b))
