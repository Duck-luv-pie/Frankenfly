# site/ -- GitHub Pages + GoDaddy DNS, for whoever has repo admin (Ducks)

`site/index.html` is the FlyBadge install page (single file, no build step). It needs to be committed as
`docs/index.html` on the default branch of `github.com/Duck-luv-pie/hunting-fly` -- that repo, not this
one, and only someone with admin on it can flip the Pages switch.

## 1. Turn on Pages (repo admin only)
1. On `hunting-fly`: copy this file to `docs/index.html`, commit, push to `main`.
2. Repo Settings -> Pages -> Source: **Deploy from a branch** -> branch `main`, folder `/docs` -> Save.
3. Wait ~1 minute, then `https://duck-luv-pie.github.io/hunting-fly/` should serve the page. Confirm
   before touching DNS.

## 2. Register the domain at GoDaddy
Candidates, cheapest/available first: `swatme.tech`, `flybrain.tech` (backups from the runbook:
`connectome.bot`, `15000neurons.com`). Any works the same way below.

## 3. Point DNS at GitHub Pages
All four candidates are apex domains (no `www.`), so this needs **A records**, not a CNAME. In GoDaddy:
DNS Management -> Records -> add four **A** records, host `@`, pointing at:
```
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153
```
Optional, if GoDaddy offers AAAA: `2606:50c0:8000::153`, `2606:50c0:8001::153`, `2606:50c0:8002::153`,
`2606:50c0:8003::153`. Delete GoDaddy's default parked-domain A record first, it conflicts.

## 4. Tell GitHub about the domain
Repo Settings -> Pages -> Custom domain -> enter the domain -> Save (this commits a `CNAME` file into
`docs/`, that's normal). Once DNS propagates (minutes to an hour), tick **Enforce HTTPS**.

Verify: `dig +short <domain>` should return the four IPs above; the page should load over `https://`.
