const tableOfContentsScript = String.raw`
function setupTableOfContentsTracking() {
  const tocLinks = Array.from(document.querySelectorAll(
    ".sidebar.right > .toc a[data-for], [data-mobile-toc] a[data-for]",
  ))
  if (tocLinks.length === 0) return

  const headingIds = new Set(tocLinks.map((link) => link.getAttribute("data-for")))
  const headings = Array.from(document.querySelector("article")?.querySelectorAll(
    "h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]",
  ) ?? []).filter((heading) => headingIds.has(heading.id))
  const desktopToc = document.querySelector(".sidebar.right > .toc .toc-content")
  const desktopHeader = document.querySelector(".sidebar.right > .toc .toc-header")
  const desktopMedia = window.matchMedia("(min-width: 1200px)")
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)")
  let activeId
  let activeFrame
  let revealRequested = true

  const revealDesktopHeading = () => {
    if (!desktopMedia.matches || !desktopToc || desktopToc.classList.contains("collapsed")) return
    const link = Array.from(desktopToc.querySelectorAll("a[data-for]")).find(
      (link) => link.getAttribute("data-for") === activeId,
    )
    if (!link || desktopToc.clientHeight === 0) return

    const containerRect = desktopToc.getBoundingClientRect()
    const linkRect = link.getBoundingClientRect()
    const top = containerRect.top + desktopToc.clientTop
    const bottom = top + desktopToc.clientHeight
    const padding = 12
    if (linkRect.top >= top + padding && linkRect.bottom <= bottom - padding) return

    // Scroll only the outline: scrollIntoView can also move the article viewport.
    const target = desktopToc.scrollTop + linkRect.top - top
      - (desktopToc.clientHeight - linkRect.height) / 2
    desktopToc.scrollTo({
      top: Math.max(0, Math.min(target, desktopToc.scrollHeight - desktopToc.clientHeight)),
      behavior: reducedMotion.matches ? "instant" : "smooth",
    })
  }

  const updateActiveHeading = () => {
    activeFrame = undefined
    let nextId = headings[0]?.id
    for (const heading of headings) {
      if (heading.getBoundingClientRect().top > 112) break
      nextId = heading.id
    }

    const changed = nextId !== activeId
    if (changed) {
      activeId = nextId
      for (const link of tocLinks) {
        const active = link.getAttribute("data-for") === activeId
        link.classList.toggle("is-active", active)
        if (active) link.setAttribute("aria-current", "location")
        else link.removeAttribute("aria-current")
      }
    }

    if (activeId && (changed || revealRequested)) revealDesktopHeading()
    revealRequested = false
  }

  const scheduleActiveHeading = () => {
    if (activeFrame !== undefined) return
    activeFrame = requestAnimationFrame(updateActiveHeading)
  }
  const requestReveal = () => {
    revealRequested = true
    scheduleActiveHeading()
  }

  window.addEventListener("scroll", scheduleActiveHeading, { passive: true })
  window.addEventListener("resize", requestReveal)
  desktopHeader?.addEventListener("click", requestReveal)
  scheduleActiveHeading()

  window.addCleanup(() => {
    if (activeFrame !== undefined) cancelAnimationFrame(activeFrame)
    window.removeEventListener("scroll", scheduleActiveHeading)
    window.removeEventListener("resize", requestReveal)
    desktopHeader?.removeEventListener("click", requestReveal)
  })
}

document.addEventListener("nav", setupTableOfContentsTracking)
`

export default tableOfContentsScript
