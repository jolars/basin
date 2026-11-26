import DefaultTheme from "vitepress/theme";
import { h } from "vue";
import AskAi from "../../src/lib/docs/AskAi.vue";
import "katex/dist/katex.min.css";
import "../../src/app.css";

export default {
    extends: DefaultTheme,
    Layout: () =>
        h(DefaultTheme.Layout, null, {
            "nav-bar-content-before": () => h(AskAi),
        }),
};
