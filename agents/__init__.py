from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.dfp import DFPAgent
from agents.dfm import DFMAgent
from agents.dsrl import DSRLAgent
from agents.grpo import GRPOAgent
from agents.svgd import SVGDAgent
from agents.mfp import MFPAgent
from agents.qam import QAMAgent
from agents.stdfp import STDFPAgent
from agents.trqam import TRQAMAgent
from agents.mdfp import MDFPAgent
from agents.qflow import QFlowAgent
from agents.rebrac import ReBRACAgent
from agents.rql import RQLAgent
from agents.anq import ANQAgent
from agents.anq2 import ANQ2Agent
from agents.anq_dfp import ANQDFPAgent
from agents.anq_rfs import ANQRFSAgent
from agents.anq_stdfp import ANQSTDFPAgent
from agents.anq_stdfp2 import ANQSTDFP2Agent
from agents.anq_stdfp3 import ANQSTDFP3Agent
from agents.mani_stdfp import ManiSTDFPAgent
from agents.dual_mani_stdfp import DualManiSTDFPAgent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    dfp=DFPAgent,
    dfm=DFMAgent,
    dsrl=DSRLAgent,
    grpo=GRPOAgent,
    svgd=SVGDAgent,
    mfp=MFPAgent,
    qam=QAMAgent,
    stdfp=STDFPAgent,
    trqam=TRQAMAgent,
    mdfp=MDFPAgent,
    rql=RQLAgent,
    rebrac=ReBRACAgent,
    qflow=QFlowAgent,
    anq=ANQAgent,
    anq2=ANQ2Agent,
    anq_rfs=ANQRFSAgent,
    anq_dfp=ANQDFPAgent,
    anq_stdfp=ANQSTDFPAgent,
    anq_stdfp2=ANQSTDFP2Agent,
    anq_stdfp3=ANQSTDFP3Agent,
    mani_stdfp=ManiSTDFPAgent,
    dual_mani_stdfp=DualManiSTDFPAgent,
)
