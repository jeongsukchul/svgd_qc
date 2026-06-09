from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.dfp import DFPAgent
from agents.svgd import SVGDAgent
from agents.mfp import MFPAgent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    dfp=DFPAgent,
    svgd=SVGDAgent,
    mfp=MFPAgent,
)
