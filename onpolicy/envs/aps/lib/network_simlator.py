import time
import torch
from channel_manager import NlosChannelManager
from aps_utils import set_random_seed
from data_store import DataStore


torch.set_printoptions(precision=20)
class NetworkSimulator:
    def __init__(self, conf):
        self.scenario_conf = conf
        self.number_of_aps = self.scenario_conf.number_of_aps
        self.number_of_ues = self.scenario_conf.number_of_ues
        self.seed = self.scenario_conf.seed
        self.step_length = self.scenario_conf.step_length
        self.tpdv = dict(device=conf.device_sim, type=conf.float_dtype_sim)
        self.serving_mask = torch.zeros((self.number_of_aps, self.number_of_ues), 
                                        dtype=torch.bool, device=conf.device_sim)
        self.datastore = DataStore(self.step_length,
                                            ['channel_coef', 'power_coef',
                                             'sinr', 'embedding',
                                             'ap_circuit_power_consumption',
                                             'transmission_power_consumption',
                                             'graph', 'clean_sinr'])
        if self.scenario_conf.precoding_algorithm == "olp":
            from power_control import OlpGnnPowerControl
            self.power_control = OlpGnnPowerControl(self.scenario_conf)
        elif self.scenario_conf.precoding_algorithm == "mrt" or self.scenario_conf.precoding_algorithm == "optimal":
            from power_control import MrtPowerControl
            self.power_control = MrtPowerControl(self.scenario_conf)
        else:
            raise NotImplementedError()
        self.channel_manager = NlosChannelManager(self.scenario_conf)

    def set_seed(self, seed):
        self.seed = seed

    def reset(self):
        self.seed += 1
        set_random_seed(self.seed)
        self.measurement_mask = self.channel_manager.reset()
        self.step(self.measurement_mask)

    def step(self, connection_choices):
        self.serving_mask = connection_choices.reshape(
            (self.number_of_aps,
             self.number_of_ues)).to(self.tpdv['device'])
        self.serving_mask *= self.measurement_mask
        # self.serving_mask = torch.tensor([[0, 0, 0, 0, 0, 1],
        #                                 [1, 0, 1, 1, 0, 0],
        #                                 [0, 0, 0, 1, 0, 0],
        #                                 [0, 0, 0, 0, 1, 1],
        #                                 [0, 0, 0, 1, 0, 0],
        #                                 [1, 0, 1, 1, 0, 1],
        #                                 [0, 1, 1, 1, 1, 0],
        #                                 [1, 1, 0, 0, 1, 0],
        #                                 [1, 1, 0, 0, 0, 0],
        #                                 [0, 1, 0, 0, 1, 1],
        #                                 [0, 0, 0, 0, 0, 0],
        #                                 [0, 0, 0, 0, 0, 1],
        #                                 [1, 0, 1, 1, 0, 0],
        #                                 [0, 0, 0, 0, 1, 1],
        #                                 [1, 1, 1, 1, 1, 0],
        #                                 [0, 0, 0, 1, 0, 0],
        #                                 [1, 1, 1, 0, 1, 1],
        #                                 [1, 1, 1, 0, 0, 0],
        #                                 [0, 1, 1, 0, 1, 0],
        #                                 [0, 0, 0, 0, 0, 0]]).to(self.serving_mask)
        for _ in range(self.step_length):
            # simulator should know everything!! => calculating channel coef with full obsevability
            G, masked_G, rho_d = self.channel_manager.step()  # adding small-scale measurements
            # masked_G = G = torch.tensor([[-1.00778991342545305771e-06-6.30550545503104863878e-07j,
            #                 1.08197948455439686689e-07+5.48621800350324783348e-08j,
            #                 8.67745325710079425976e-08-8.06480871961855028633e-08j,
            #                 -2.85495547502159375468e-08+6.01833059899452178946e-08j,
            #                 -1.12170437477643759866e-07+3.77896567257246979802e-07j,
            #                 3.09574800555570015355e-08-1.29563861098969242033e-08j],
            #                 [ 9.19765283917738862207e-08-1.24867137642109124516e-07j,
            #                 3.83330847521280150233e-07-1.71327658567588215775e-07j,
            #                 -1.27669878923231884158e-06+8.68058911622132384579e-07j,
            #                 1.46060131204737554290e-06+8.67839320596942594212e-08j,
            #                 -2.61294690077220551349e-07+2.40074061493456903095e-07j,
            #                 2.68437479077940657146e-08+1.58478530131073440013e-07j],
            #                 [-2.56480439341500108403e-07+3.28871022467540806882e-07j,
            #                 -8.84596205845944051306e-07+5.61932171769673312788e-07j,
            #                 1.29283074049228162687e-07-6.55508409054200444312e-08j,
            #                 1.28981879017321008137e-06-1.96497899709491573089e-06j,
            #                 1.36781542111852583121e-08+2.29749666910837089035e-07j,
            #                 -1.32320312848652027438e-08-4.14786050448367154638e-10j],
            #                 [-2.25944554981503233361e-06-4.44849312785232690952e-06j,
            #                 4.19117653058614805542e-07+1.08535365760333883411e-08j,
            #                 -4.85308508427306188576e-07+6.40184540211284067538e-07j,
            #                 3.11457829323198242826e-08-8.90646423929906193446e-08j,
            #                 5.32305828972060714709e-08+1.07193409157480519486e-06j,
            #                 2.59927262946353357560e-05+3.25139148586723959313e-06j],
            #                 [ 5.83403584565915285740e-07-2.56944957015314466640e-07j,
            #                 -3.45328177377074819622e-06-2.27753141185482726129e-06j,
            #                 7.58094978576426473830e-08-1.46209239281856383221e-06j,
            #                 -6.84866280047533870654e-07+1.16355642845302909744e-06j,
            #                 4.50679312934666448574e-07+3.07943954159602010782e-07j,
            #                 1.19404129562695176742e-07-2.67889486617482399826e-07j],
            #                 [ 4.30167402129482583781e-07-1.18180583393377811184e-07j,
            #                 4.86492057477812458079e-07+1.91787058441886183713e-08j,
            #                 5.94090233136393896726e-07+1.77737810322939727311e-06j,
            #                 -9.44550358857204250402e-08-1.52698657887910390905e-06j,
            #                 -2.01457029020094428569e-07+5.41921811485853826612e-09j,
            #                 2.89580733738591211783e-08-6.14069081480183130440e-07j],
            #                 [ 2.28246013808039691560e-07+9.05918557174098921989e-08j,
            #                 -4.58972588558275957445e-06+5.88847495582077442390e-07j,
            #                 3.00351906948107554076e-08-1.84750326943164491240e-07j,
            #                 -1.47209107523924909187e-06+3.67982247296474691503e-06j,
            #                 -5.30087134771027751400e-08-3.17216148099357633696e-07j,
            #                 1.53576623608755625871e-07+4.09406981864088093083e-08j],
            #                 [-4.06884121990484562477e-07-6.95591799275196198353e-07j,
            #                 -8.88902386782849818197e-07-5.51789082183098822382e-08j,
            #                 1.19931679794326363566e-08-4.65704965060595577923e-07j,
            #                 6.34567785984051705827e-07+3.91189842888370279494e-07j,
            #                 -2.39078274848890311639e-07+5.98157132249019222462e-09j,
            #                 1.15758616907283837922e-07+1.78041165798375677839e-08j],
            #                 [-2.33556510024976606936e-08-1.80322787029015445511e-07j,
            #                 -3.61217254574987186150e-07-2.29131945734285611526e-07j,
            #                 -6.61093288203909803226e-08-6.89480610966198564666e-08j,
            #                 1.13984331824869727760e-07+4.06413373907056511799e-07j,
            #                 -9.69337982192280404919e-07-1.28531203651002600000e-07j,
            #                 -1.47501283099468176798e-07-3.83903881504570260274e-08j],
            #                 [ 3.35876388185438898899e-07-5.54583962541448982651e-07j,
            #                 -3.31904722710328276147e-06+4.51580948813493630346e-06j,
            #                 -7.30994481635055923177e-07+2.59857811404923729100e-07j,
            #                 1.09424276533893717114e-07-6.24586938883564157895e-09j,
            #                 1.64426829353503042005e-07-7.87698735669910029590e-08j,
            #                 -2.03719743168833936853e-07-1.87860670212453370811e-07j],
            #                 [-4.90426915693894390681e-08+8.67015888256130775998e-08j,
            #                 2.50285631093951604255e-07+6.65312282254363344883e-07j,
            #                 -5.36115619202319922843e-08-2.92351909624824372341e-08j,
            #                 8.46470687511814502757e-09-2.74620117434526844085e-08j,
            #                 -2.21970810602272452269e-10-1.05194977626675394314e-06j,
            #                 1.57460829925666057807e-07+6.77534482152711793807e-07j],
            #                 [ 3.07021528707727939752e-07-3.15893195109466502506e-08j,
            #                 4.94496274310218429561e-07-3.17827630663105685285e-07j,
            #                 1.20394004791134384893e-07+8.55817175826798348429e-09j,
            #                 -1.09075986925899899385e-07+3.99605329069745340914e-07j,
            #                 5.69708795492980329179e-08+1.86897153139091454057e-08j,
            #                 -3.96971572852172834433e-08+6.71349299771712671100e-08j],
            #                 [ 1.17350137351848969592e-06+1.00223949740440268060e-06j,
            #                 -7.82984840357874829840e-07+4.06110211575861277738e-07j,
            #                 1.23151999771069951627e-05+5.35271753573764392451e-06j,
            #                 1.64468088983245522199e-06+3.59057453307902841474e-07j,
            #                 4.03458400620355979177e-07-7.39116110383012563539e-07j,
            #                 -3.93310374853451921272e-08-3.09095727784885524003e-08j],
            #                 [-1.37346689942595894969e-07+1.90313831130149512538e-07j,
            #                 -4.83615474370647885147e-07+5.07772050867137987262e-07j,
            #                 -1.09768988057794700356e-08-1.27887475864398752041e-07j,
            #                 -2.84876351098934849890e-08+1.51951169577191582875e-08j,
            #                 -2.27190750892202662998e-06-1.31728190215169138978e-06j,
            #                 8.25664212907257389162e-07-8.85000539927667967963e-06j],
            #                 [ 1.35288650165844646090e-07+2.40058073111979335302e-07j,
            #                 5.11170259620745743371e-07+6.15477400901456794045e-08j,
            #                 1.15328519730509512013e-07-3.57954089121637434084e-07j,
            #                 -1.93769965282779592017e-05-4.81138354802219844449e-05j,
            #                 -2.42368154803097235396e-07-1.31007712360584008564e-07j,
            #                 2.37569324335295267814e-09-1.87578496634118576486e-08j],
            #                 [ 1.15767320701265872476e-07+1.06122253499009694749e-07j,
            #                 -3.38319118344331166530e-08+1.87304894964771461291e-07j,
            #                 -8.92985341389321453972e-08-7.17521124272849253751e-08j,
            #                 2.41701297296417433134e-09-7.53395991437098460676e-08j,
            #                 -1.10726412346023143842e-07+3.07169080230464100728e-08j,
            #                 -1.60542123114167658756e-08+1.52304035185093725382e-07j],
            #                 [ 7.27619158182136293100e-07-3.57639281324757359390e-06j,
            #                 3.85581065246854875952e-08+1.21954533691282337982e-07j,
            #                 1.75558957895246617275e-06+1.21495808465827589635e-06j,
            #                 6.19671456014581854506e-09+2.42732822067397508282e-08j,
            #                 1.25333258465607822696e-06+1.36961060044976316531e-06j,
            #                 -1.17664056533124299863e-06-8.51338600372275270810e-07j],
            #                 [-1.09431616230304702285e-06+4.69509755957230012787e-07j,
            #                 -1.07461458187382428832e-06-2.10722558679604641372e-07j,
            #                 -1.55745681775669843764e-07-2.50884814760553715461e-07j,
            #                 -2.62166479961406704629e-07-4.68726287646407707883e-07j,
            #                 2.17619971373741336725e-07-2.47624896788198001130e-07j,
            #                 -2.24405500489282087568e-07+2.77257152750499276932e-07j],
            #                 [-5.80953417703459173399e-08-2.64585580083030374318e-07j,
            #                 6.30258933164795607743e-08+2.19315516525607681705e-07j,
            #                 -4.92552913280194825098e-07-5.83678162559963951479e-07j,
            #                 -1.08212354823030776599e-07-2.45905068050742791662e-07j,
            #                 -4.52961274806225629541e-08+5.09166641653934307488e-08j,
            #                 -2.71221874604219979067e-07-8.03837668780887735378e-08j],
            #                 [-1.52770586389290160878e-08-1.60553308301067836341e-08j,
            #                 3.34228416563208108772e-07-2.94100897411606652222e-07j,
            #                 -2.03216604683485634098e-08+1.58370415003376330842e-07j,
            #                 -1.79099936761433718439e-07-1.47356223393934775437e-07j,
            #                 2.59978129885574870485e-08+1.45956762216435147288e-08j,
            #                 9.17734605941872978189e-08-1.21282309780224252149e-07j]]).to(G)
            # print("G: ", G)
            # print("rho_d: ", rho_d)
            # print("serving_mask: ", self.serving_mask)

            if self.scenario_conf.precoding_algorithm == "optimal":
                _, allocated_power = self.power_control.get_optimal_sinr(G, rho_d, self.serving_mask.clone().cpu().numpy()) # allocating power
                embedding, graph = None, None
                allocated_power = torch.tensor(allocated_power).to(G)
                # print("allocated_power: ", allocated_power)
            else:
                if self.scenario_conf.if_remove_off_aps_form_olp:
                    off_aps = (self.serving_mask == 0).all(dim=1).nonzero(as_tuple=True)[0]
                    mask = torch.ones(G.shape[0], dtype=torch.bool)
                    mask[off_aps] = False
                    G_reduced = G[mask]
                    serving_mask_reduced = self.serving_mask[mask]
                    allocated_power_reduced, embedding, graph = self.power_control.get_power_coef(G_reduced, rho_d, serving_mask_reduced, return_graph=True) # G_reduced
                    allocated_power_reduced.reshape(G_reduced.shape)
                    allocated_power = torch.zeros_like(G)
                    allocated_power[mask] = allocated_power_reduced
                else:
                    allocated_power, embedding, graph = self.power_control.get_power_coef(G, rho_d, self.serving_mask, return_graph=True)

            # to simulate aps, we set the non-activated power coef to zero
            masked_allocated_power = (allocated_power.clone().detach() * self.serving_mask).to(allocated_power)
            # calc power consumption
            transmission_power_consumption = self.power_control.get_transmission_power(masked_allocated_power)
            ap_circuit_power_consumption = self.power_control.get_ap_circuit_power(self.serving_mask)
            # calc sinr with full channel info and the maked allocated power
            sinr = self.power_control.calcualte_sinr(G, rho_d, masked_allocated_power)
            clean_sinr = self.power_control.calcualte_sinr(G, rho_d, allocated_power)
            # print("clear sinr", clean_sinr)
            # print("sinr", sinr)
            # print("---------")
            # if (sinr.std() > 3).any():
            #     raise

            # time.sleep(0.1)
            # store the info
            self.datastore.add(channel_coef=masked_G, power_coef=masked_allocated_power, 
                               embedding=embedding, sinr=sinr, clean_sinr=clean_sinr,
                               transmission_power_consumption=transmission_power_consumption,
                               ap_circuit_power_consumption=ap_circuit_power_consumption,
                               graph=graph)   # add to the data store
        # self.channel_manager.assign_measurement_aps()
