# VLALagBench project website

Static project page for [VLALagBench](https://github.com/codestrongestx/vla-lag-bench).
Site URL once enabled: https://codestrongestx.github.io/vla-lag-bench/

The website source and its media live in `website/` on `main`, alongside the
benchmark code. No framework, build tool, simulator, or external artifact
download is needed to deploy the page.

## Deployment

`.github/workflows/pages.yml` uploads only this directory to GitHub Pages when
website files or the workflow change on `main`. It can also be run manually
from the Actions tab. Deployment is skipped while the repository is private.

After the repository is made public, select **Settings → Pages → Build and
deployment → Source → GitHub Actions**, then run **Deploy project website**
from the Actions tab. The workflow does not change repository visibility.
A separate `gh-pages` branch is no longer used for deployment.

## Updating the page

Edit `index.html`, `style.css`, or `site.js` here and push to `main`. Keep local
asset URLs relative so they work under `/vla-lag-bench/`; keep the canonical
URL and Open Graph URLs absolute.

Figures and videos are exported from the paper website build in the report
repository. Preserve the paper's wording, verified rates, and recorded outcomes
when replacing them. The manuscript stays in the report repository; this site
links to the [paper on arXiv](https://arxiv.org/abs/2611.03103).

The original materials are licensed under [CC BY-NC 4.0](LICENSE).
