import { PageFrame, PageFrameProps } from "./types"
import HeaderConstructor from "../Header"
import MobileTableOfContentsConstructor from "../MobileTableOfContents"
import LeftSidebarToggleConstructor from "../LeftSidebarToggle"

const Header = HeaderConstructor()
const MobileTableOfContents = MobileTableOfContentsConstructor()
const LeftSidebarToggle = LeftSidebarToggleConstructor()

/**
 * The default page frame — three-column layout with left sidebar, center
 * content (header + body + afterBody), and right sidebar, followed by a footer.
 *
 * This is the original Quartz layout, extracted from renderPage.tsx.
 */
export const DefaultFrame: PageFrame = {
  name: "default",
  components: [MobileTableOfContents, LeftSidebarToggle],
  render({
    componentData,
    header,
    beforeBody,
    pageBody: Content,
    afterBody,
    left,
    right,
    footer,
  }: PageFrameProps) {
    return (
      <>
        {left.length > 0 && <LeftSidebarToggle {...componentData} />}
        <div class="left sidebar" id="left-sidebar">
          {left.map((BodyComponent) => (
            <BodyComponent {...componentData} />
          ))}
        </div>
        <div class="center">
          <div class="page-header">
            <Header {...componentData}>
              {header.map((HeaderComponent) => (
                <HeaderComponent {...componentData} />
              ))}
            </Header>
            <div class="popover-hint">
              {beforeBody.map((BodyComponent) => (
                <BodyComponent {...componentData} />
              ))}
            </div>
          </div>
          <Content {...componentData} />
          <hr />
          <div class="page-footer">
            {afterBody.map((BodyComponent) => (
              <BodyComponent {...componentData} />
            ))}
          </div>
        </div>
        <div class="right sidebar">
          {right.map((BodyComponent) => (
            <BodyComponent {...componentData} />
          ))}
        </div>
        <MobileTableOfContents {...componentData} />
        {footer.map((FooterComponent) => (
          <FooterComponent {...componentData} />
        ))}
      </>
    )
  },
}
