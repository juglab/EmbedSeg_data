from datasets.DSB2018Dataset import DSB2018Dataset
from datasets.DSBReduced2018Dataset import DSBReduced2018Dataset
from datasets.MouseNuclei2020Dataset import MouseNuclei2020Dataset
from datasets.MouseNucleiElastic2020Dataset import MouseNucleiElastic2020Dataset
from datasets.Usiigaci2019Dataset import Usiigaci2019Dataset
from datasets.CTC2017Dataset import CTC2017Dataset
from datasets.Elegans2012Dataset import Elegans2012Dataset
from datasets.CVPPP2014Dataset import CVPPP2014Dataset


def get_dataset(name, dataset_opts):
    if name == "dsb2018": 
        return DSB2018Dataset(**dataset_opts)
    elif name=="dsbreduced2018":
        return DSBReduced2018Dataset(**dataset_opts)
    elif name=="mousenuclei2020":
        return MouseNuclei2020Dataset(**dataset_opts)
    elif name=="mousenucleielastic2020":
        return MouseNucleiElastic2020Dataset(**dataset_opts)
    elif name=="usiigaci2019":
        return Usiigaci2019Dataset(**dataset_opts)
    elif name == "ctc2017":
        return CTC2017Dataset(**dataset_opts)
    elif name=="elegans2012":
        return Elegans2012Dataset(**dataset_opts)
    elif name=="cvppp2014":
        return CVPPP2014Dataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))
