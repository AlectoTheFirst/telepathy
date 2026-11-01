"""8.3 Intent-probe accuracy with and without the session key. Keyed DATA is AES-GCM per packet:
with the escrowed key the probe decrypts; without it, it reads the ciphertext as slots."""
from ..config import DATA_INTENTS
from ..models.evaluate import send
from ..otel.probe import load_probe
from .common import CAT, DEFAULT, figure, models, save, savefig
from .e6_monitor_tap import seal_messages


def main(n=3000):
    m, meta, td = models(DEFAULT)
    clf, _ = load_probe(DEFAULT)
    ev = td.eval_set(n, intents=DATA_INTENTS)
    slots, mask = send(m["A"], ev)
    cipher, plain = seal_messages(slots, 9)            # fresh key per message (session)
    acc_key = float((clf.classify(plain, mask)[0] == ev.intent).float().mean())      # escrowed key: decrypted
    acc_nokey = float((clf.classify(cipher, mask)[0] == ev.intent).float().mean())
    print(f"with key {acc_key:.3f}, without key {acc_nokey:.3f} (chance 0.333)")
    save("t3_probe_key", {"with_key": acc_key, "without_key": acc_nokey, "chance": 1 / 3,
                          "cipher": "AES-256-GCM per DATA packet"})
    fig, ax = figure(4.4, 3.4)
    ax.bar([0, 1], [acc_key, acc_nokey], color=[CAT[0], CAT[1]], width=0.5)
    for i, v in enumerate((acc_key, acc_nokey)):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(1 / 3, color="#8a8a85", ls=":", lw=1)
    ax.set_xticks([0, 1], ["with session key\n(escrowed)", "without key\n(ciphertext)"], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("intent accuracy")
    ax.set_title("Probe needs the escrowed key", fontsize=9)
    savefig(fig, "t3_probe_key")


if __name__ == "__main__":
    main()
