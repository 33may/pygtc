from loguru import logger
from matplotlib import pyplot as plt


def plot_spread(stock1, stock2, spread = None, alpha= None, beta= None, stock1_name = None, stock2_name = None):

    alpha = 0 if alpha is None else alpha
    stock1_name = stock1_name if stock1_name else "Stock 1"
    stock2_name = stock2_name if stock2_name else "Stock 2"

    spread_values = None

    if spread:
        spread_values = spread_values
    elif beta:
        spread_values =  stock1 - (alpha + beta * stock2)
    else:
        logger.error(f"spread and alpha and beta are both None")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{stock1_name} vs {stock2_name} — Price & Spread", fontsize=14)

    ax1.plot(stock1, label=stock1_name)
    ax1.plot(stock2, label=stock2_name, linestyle="--", alpha=0.5)
    ax1.plot(alpha + beta * stock2, label=f"{stock2_name} (Transformed)")
    ax1.legend()
    ax1.set_ylabel("Price")
    if (alpha is not None) and (beta is not None):
        txt = rf"$\alpha$ = {alpha:.4g},  $\beta$ = {beta:.4g}"
    else:
        txt = "α/β: n/a"

    ax1.text(
        0.02, 0.98, txt,
        transform=ax1.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )


    ax2.plot(spread_values, label="Spread")
    ax2.axhline(y=spread_values.mean(), linestyle="--", color="red", label="Mean")
    ax2.legend()
    ax2.set_ylabel("Spread")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.show()